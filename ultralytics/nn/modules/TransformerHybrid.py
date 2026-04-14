"""
Transformer Hybrid Backbone Modules for YOLO11 Integration
============================================================
Three state-of-the-art Transformer architectures adapted as YOLO backbone stages:

1. **Swin Transformer** (ICCV 2021 Best Paper):
   - Window-based multi-head self-attention (W-MSA)
   - Shifted window for cross-window connections (SW-MSA)
   - Hierarchical representation with patch merging
   - Linear complexity O(n) vs O(n²) for global attention

2. **ViT (Vision Transformer)** hybrid:
   - Conv stem for robust early features (no pure patch embedding)
   - Standard multi-head self-attention for global context
   - Learnable 2D positional embeddings for spatial awareness
   - Conv projection back to spatial feature maps

3. **Mobile-Former** (CVPR 2022):
   - Bidirectional bridge between MobileNet (local) and Transformer (global)
   - Mobile branch: lightweight depthwise separable convolutions
   - Former branch: small set of learnable global tokens
   - Mobile→Former: spatial features enrich global tokens
   - Former→Mobile: global context enhances local features
   - Best of both worlds: local detail + global reasoning

Architecture stages for YOLO compatibility:
- SwinStage: Swin Transformer blocks with optional patch merging (P3-P5)
- ViTStage: ViT blocks with conv stem for early/mid stages
- MobileFormerStage: Bidirectional conv+transformer hybrid
- FeatureAlignTF: Channel alignment for backbone→neck transition
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# ────────────────────────────────────────────────────────────────────────
# DROP PATH (STOCHASTIC DEPTH)
# ────────────────────────────────────────────────────────────────────────
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Apply stochastic depth per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# ════════════════════════════════════════════════════════════════════════
# SWIN TRANSFORMER — Window-based attention with shifted windows
# ════════════════════════════════════════════════════════════════════════

class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention (W-MSA) from Swin Transformer.

    Computes self-attention within local windows of fixed size, achieving
    linear complexity O(n) instead of O(n²) for the full feature map.
    Relative position bias provides spatial awareness within each window.

    Args:
        dim: Number of input channels
        window_size: Window size (H_w, W_w) — attention computed within this area
        num_heads: Number of attention heads
        qkv_bias: Add learnable bias to Q, K, V projections
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table: (2*Wh-1) * (2*Ww-1) entries, num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index for each token pair in the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C) where N = window_size[0] * window_size[1]
            mask: (num_windows, N, N) or None for W-MSA, provided for SW-MSA
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, N
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply mask for shifted window attention
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinMLP(nn.Module):
    """MLP block for Swin Transformer with GELU activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Single Swin Transformer Block with W-MSA or SW-MSA.

    Alternates between regular window attention (W-MSA) and shifted window
    attention (SW-MSA) to enable cross-window connections.

    Architecture: LN → W-MSA/SW-MSA → residual → LN → MLP → residual

    Args:
        dim: Number of input channels
        num_heads: Number of attention heads
        window_size: Window size for local attention
        shift_size: Shift size for SW-MSA (0 for W-MSA, window_size//2 for SW-MSA)
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        drop_path: Stochastic depth rate
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def _compute_attn_mask(self, H, W, device):
        """Compute attention mask for shifted window attention."""
        if self.shift_size == 0:
            return None

        # Calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Window partition the mask
        mask_windows = self._window_partition(img_mask, self.window_size)  # nW, Wh*Ww, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    @staticmethod
    def _window_partition(x, window_size):
        """Partition feature map into non-overlapping windows."""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        return windows

    @staticmethod
    def _window_reverse(windows, window_size, H, W):
        """Reverse window partition back to feature map."""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) — standard conv feature map format
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Pad feature map to be divisible by window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        _, _, Hp, Wp = x.shape

        # Reshape to (B, Hp, Wp, C) for window operations
        x = x.permute(0, 2, 3, 1).contiguous()

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Compute attention mask
        attn_mask = self._compute_attn_mask(Hp, Wp, x.device)

        # Window partition
        x_windows = self._window_partition(shifted_x, self.window_size)

        # W-MSA / SW-MSA
        shortcut = x_windows
        x_windows = self.norm1(x_windows)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = shortcut + self.drop_path(attn_windows)

        # MLP
        attn_windows = attn_windows + self.drop_path(self.mlp(self.norm2(attn_windows)))

        # Reverse window partition
        shifted_x = self._window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # Back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging for Swin Transformer — spatial downsampling (2x).

    Concatenates 2×2 neighboring patches then projects to target channels.
    This is how Swin creates its hierarchical multi-scale representation.

    Args:
        c1: Input channels
        c2: Output channels (typically 2*c1)
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.reduction = nn.Linear(4 * c1, c2, bias=False)
        self.norm = nn.LayerNorm(4 * c1)

    def forward(self, x):
        """(B, C, H, W) → (B, c2, H/2, W/2)"""
        B, C, H, W = x.shape

        # Pad if not divisible by 2
        if H % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))
        if W % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0))

        B, C, H, W = x.shape
        # Reshape to (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()

        # Gather 2×2 patches
        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C

        x = self.norm(x)
        x = self.reduction(x)

        # Back to (B, C_out, H/2, W/2)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SwinStage(nn.Module):
    """
    Swin Transformer Stage for YOLO backbone.

    Contains optional PatchMerging (stride=2) + n Swin Transformer blocks.
    Blocks alternate between W-MSA and SW-MSA for cross-window connections.

    Key advantages:
    - Linear complexity O(n) through window-based attention
    - Hierarchical multi-scale features through patch merging
    - Cross-window connections via shifted windows
    - Strong local + non-local feature modeling

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of Swin Transformer blocks
        s: Stride (1 = no downsampling, 2 = patch merging)
        num_heads: Number of attention heads (auto if 0)
        window_size: Window size for local attention
        mlp_ratio: MLP expansion ratio
        drop_path: Maximum stochastic depth probability
    """
    def __init__(self, c1, c2, n=2, s=1, num_heads=0, window_size=7,
                 mlp_ratio=4., drop_path=0.1):
        super().__init__()

        # Auto-determine num_heads: ensure c2 is divisible
        if num_heads == 0:
            num_heads = max(2, min(8, c2 // 32))
            while c2 % num_heads != 0 and num_heads > 1:
                num_heads -= 1

        # Downsampling via PatchMerging (when stride=2)
        if s == 2:
            self.downsample = PatchMerging(c1, c2)
        elif c1 != c2:
            self.downsample = nn.Sequential(
                nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2),
            )
        else:
            self.downsample = nn.Identity()

        # Stack of Swin Transformer blocks with alternating shift
        blocks = []
        for i in range(n):
            dp = drop_path * (1 - math.cos(math.pi * i / max(n - 1, 1))) / 2 if n > 1 else drop_path
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            blocks.append(SwinTransformerBlock(
                dim=c2,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                drop_path=dp,
            ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


# ════════════════════════════════════════════════════════════════════════
# VIT HYBRID — Conv stem + ViT blocks for global self-attention
# ════════════════════════════════════════════════════════════════════════

class ViTSelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention for ViT.

    Full global attention: every spatial position attends to every other.
    More expensive than Swin's window attention but captures longer-range
    dependencies directly.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        attn_drop: Attention dropout
        proj_drop: Output projection dropout
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: (B, N, C) → (B, N, C)"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(nn.Module):
    """
    Single ViT Transformer Block.

    Architecture: LN → MHSA → residual → LN → MLP → residual

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        drop_path: Stochastic depth rate
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ViTSelfAttention(dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        """x: (B, N, C) → (B, N, C)"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTStage(nn.Module):
    """
    ViT Hybrid Stage for YOLO backbone.

    Uses conv-based patch embedding for downsampling (more robust than
    pure linear projection) and standard ViT blocks for global attention.
    Learnable 2D positional embeddings provide spatial awareness.

    Key advantages:
    - Global receptive field from the first block
    - Strong for capturing long-range dependencies
    - Conv stem preserves local spatial structure
    - Simple and well-studied architecture

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of ViT blocks
        s: Stride (1 or 2, controls conv patch embedding)
        num_heads: Number of attention heads (auto if 0)
        mlp_ratio: MLP expansion ratio
        drop_path: Maximum stochastic depth probability
    """
    def __init__(self, c1, c2, n=2, s=1, num_heads=0, mlp_ratio=4., drop_path=0.1):
        super().__init__()

        # Auto-determine num_heads
        if num_heads == 0:
            num_heads = max(2, min(8, c2 // 32))
            while c2 % num_heads != 0 and num_heads > 1:
                num_heads -= 1

        # Conv-based patch embedding (handles downsampling)
        if s == 2:
            self.patch_embed = nn.Sequential(
                nn.Conv2d(c1, c2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )
        elif c1 != c2:
            self.patch_embed = nn.Sequential(
                nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )
        else:
            self.patch_embed = nn.Identity()

        # ViT blocks
        blocks = []
        for i in range(n):
            dp = drop_path * (1 - math.cos(math.pi * i / max(n - 1, 1))) / 2 if n > 1 else drop_path
            blocks.append(ViTBlock(
                dim=c2,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dp,
            ))
        self.blocks = nn.Sequential(*blocks)

        # Layer norm after all blocks
        self.norm = nn.LayerNorm(c2)

        # Positional embedding — will be lazily initialized on first forward
        self.pos_embed = None
        self.c2 = c2

    def _get_pos_embed(self, H, W, device):
        """Get or create 2D positional embedding."""
        if self.pos_embed is None or self.pos_embed.shape[1] != H * W:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, H * W, self.c2, device=device)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        return self.pos_embed

    def forward(self, x):
        """(B, C, H, W) → (B, c2, H', W')"""
        x = self.patch_embed(x)
        B, C, H, W = x.shape

        # Reshape to tokens: (B, C, H, W) → (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embedding
        if self.training or self.pos_embed is None:
            pos = self._get_pos_embed(H, W, x.device)
            # Interpolate if size changed
            if pos.shape[1] != H * W:
                pos = self._get_pos_embed(H, W, x.device)
            x = x + pos
        elif self.pos_embed is not None and self.pos_embed.shape[1] == H * W:
            x = x + self.pos_embed

        # Apply ViT blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Reshape back to spatial: (B, H*W, C) → (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


# ════════════════════════════════════════════════════════════════════════
# MOBILE-FORMER — Bidirectional bridge between Conv and Transformer
# ════════════════════════════════════════════════════════════════════════

class MobileBlock(nn.Module):
    """
    Mobile branch block — lightweight depthwise separable convolution.

    Efficient local feature extraction using MobileNet-style operations:
    DW conv (spatial) → BN → SiLU → PW conv (channel) → BN → SiLU

    Args:
        dim: Number of channels
        expand: Expansion ratio for intermediate channels
        kernel_size: Kernel size for depthwise conv
    """
    def __init__(self, dim, expand=4, kernel_size=3):
        super().__init__()
        hidden = int(dim * expand)
        self.dw = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.act1 = nn.SiLU(inplace=True)
        self.pw1 = nn.Conv2d(dim, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.act2 = nn.SiLU(inplace=True)
        self.pw2 = nn.Conv2d(hidden, dim, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw1(x)))
        x = self.bn3(self.pw2(x))
        return x + residual


class FormerBlock(nn.Module):
    """
    Former (Transformer) branch block — global token self-attention.

    Processes a small set of learnable global tokens using standard
    multi-head self-attention. These tokens capture global context
    that gets bridged back to the mobile (conv) branch.

    Args:
        dim: Token dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        drop_path: Stochastic depth rate
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=2., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ViTSelfAttention(dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, tokens):
        """tokens: (B, M, D) → (B, M, D) where M = number of global tokens"""
        tokens = tokens + self.drop_path(self.attn(self.norm1(tokens)))
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        return tokens


class Mobile2Former(nn.Module):
    """
    Mobile → Former bridge: spatial features enrich global tokens.

    Cross-attention where global tokens (Q) attend to spatial features (K, V).
    This allows the transformer branch to absorb local spatial information
    from the conv branch.

    Args:
        dim: Feature dimension (must match both branches)
        num_heads: Number of cross-attention heads
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_tokens = nn.LayerNorm(dim)
        self.norm_spatial = nn.LayerNorm(dim)

    def forward(self, tokens, spatial):
        """
        Args:
            tokens: (B, M, D) — global tokens (will be updated)
            spatial: (B, C, H, W) — spatial features from mobile branch
        Returns:
            updated tokens: (B, M, D)
        """
        B, M, D = tokens.shape
        _, C, H, W = spatial.shape

        # Flatten spatial: (B, C, H, W) → (B, H*W, C)
        spatial_flat = spatial.flatten(2).transpose(1, 2)

        # Cross-attention: tokens attend to spatial features
        q = self.q_proj(self.norm_tokens(tokens))  # (B, M, D)
        kv = self.kv_proj(self.norm_spatial(spatial_flat))  # (B, HW, 2D)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(B, M, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, H * W, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, H * W, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, M, D)

        return tokens + self.out_proj(out)


class Former2Mobile(nn.Module):
    """
    Former → Mobile bridge: global context enhances spatial features.

    Cross-attention where spatial features (Q) attend to global tokens (K, V).
    This injects global reasoning from the transformer branch back into
    the conv branch's local features.

    Args:
        dim: Feature dimension
        num_heads: Number of cross-attention heads
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_spatial = nn.LayerNorm(dim)
        self.norm_tokens = nn.LayerNorm(dim)

    def forward(self, spatial, tokens):
        """
        Args:
            spatial: (B, C, H, W) — spatial features (will be updated)
            tokens: (B, M, D) — global tokens from former branch
        Returns:
            updated spatial: (B, C, H, W)
        """
        B, C, H, W = spatial.shape
        _, M, D = tokens.shape

        # Flatten spatial
        spatial_flat = spatial.flatten(2).transpose(1, 2)  # (B, HW, C)

        # Cross-attention: spatial attends to tokens
        q = self.q_proj(self.norm_spatial(spatial_flat))  # (B, HW, D)
        kv = self.kv_proj(self.norm_tokens(tokens))  # (B, M, 2D)
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B, H * W, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, M, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, D)

        out = self.out_proj(out)

        # Reshape back to spatial and add residual
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return spatial + out


class MobileFormerUnit(nn.Module):
    """
    Single Mobile-Former unit: one round of bidirectional information exchange.

    Flow: Mobile block → Mobile→Former bridge → Former block → Former→Mobile bridge

    This interleaved pattern ensures both branches continuously benefit from
    each other's complementary strengths (local details + global context).

    Args:
        dim: Feature dimension (shared across both branches)
        num_tokens: Number of global tokens
        num_heads: Number of attention heads
        mobile_expand: Expansion ratio for mobile block
        mlp_ratio: MLP expansion ratio for former block
        drop_path: Stochastic depth rate
    """
    def __init__(self, dim, num_tokens=6, num_heads=4, mobile_expand=4,
                 mlp_ratio=2., drop_path=0.):
        super().__init__()
        self.mobile_block = MobileBlock(dim, expand=mobile_expand)
        self.m2f = Mobile2Former(dim, num_heads=num_heads)
        self.former_block = FormerBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path)
        self.f2m = Former2Mobile(dim, num_heads=num_heads)

    def forward(self, spatial, tokens):
        """
        Args:
            spatial: (B, C, H, W) — spatial features
            tokens: (B, M, D) — global tokens
        Returns:
            updated spatial, updated tokens
        """
        # 1. Mobile block: local feature processing
        spatial = self.mobile_block(spatial)

        # 2. Mobile → Former: spatial features enrich tokens
        tokens = self.m2f(tokens, spatial)

        # 3. Former block: global token self-attention
        tokens = self.former_block(tokens)

        # 4. Former → Mobile: global context enhances spatial
        spatial = self.f2m(spatial, tokens)

        return spatial, tokens


class MobileFormerStage(nn.Module):
    """
    Mobile-Former Stage for YOLO backbone.

    Combines MobileNet-style conv branch with Transformer branch connected
    through bidirectional bridges. A small set of learnable global tokens
    accumulates global information and distributes it back to spatial features.

    Key advantages:
    - Bidirectional information flow between local and global branches
    - Very efficient: only M tokens (M<<HW) in transformer branch
    - MobileNet branch handles spatial details cheaply
    - Global tokens capture scene-level context
    - Complementary strengths: local detail + global reasoning

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of Mobile-Former units
        s: Stride (1 or 2, via conv downsampling)
        num_tokens: Number of learnable global tokens
        num_heads: Number of attention heads (auto if 0)
        drop_path: Maximum stochastic depth probability
    """
    def __init__(self, c1, c2, n=2, s=1, num_tokens=6, num_heads=0, drop_path=0.1):
        super().__init__()

        # Auto-determine num_heads
        if num_heads == 0:
            num_heads = max(2, min(8, c2 // 32))
            while c2 % num_heads != 0 and num_heads > 1:
                num_heads -= 1

        # Downsampling via strided conv (when stride=2)
        if s == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(c1, c2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )
        elif c1 != c2:
            self.downsample = nn.Sequential(
                nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )
        else:
            self.downsample = nn.Identity()

        # Learnable global tokens
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, c2) * 0.02)

        # Stack of Mobile-Former units
        units = []
        for i in range(n):
            dp = drop_path * (1 - math.cos(math.pi * i / max(n - 1, 1))) / 2 if n > 1 else drop_path
            units.append(MobileFormerUnit(
                dim=c2,
                num_tokens=num_tokens,
                num_heads=num_heads,
                mobile_expand=4,
                mlp_ratio=2.,
                drop_path=dp,
            ))
        self.units = nn.ModuleList(units)

    def forward(self, x):
        """(B, C, H, W) → (B, c2, H', W')"""
        x = self.downsample(x)
        B = x.shape[0]

        # Expand tokens for batch
        tokens = self.tokens.expand(B, -1, -1)

        # Run Mobile-Former units
        for unit in self.units:
            x, tokens = unit(x, tokens)

        return x


# ════════════════════════════════════════════════════════════════════════
# FEATURE ALIGNMENT — Backbone → Neck channel bridging
# ════════════════════════════════════════════════════════════════════════

class FeatureAlignTF(nn.Module):
    """
    Feature alignment for Transformer backbone → YOLO neck channel calibration.

    Simple 1×1 conv + BN + SiLU when channels don't match, Identity otherwise.

    Args:
        c1: Input channels
        c2: Output channels
    """
    def __init__(self, c1, c2):
        super().__init__()
        if c1 != c2:
            self.align = nn.Sequential(
                nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )
        else:
            self.align = nn.Identity()

    def forward(self, x):
        return self.align(x)
