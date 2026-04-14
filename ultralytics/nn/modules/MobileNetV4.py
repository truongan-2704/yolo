"""
MobileNetV4 — Universal Backbone Modules for YOLO Integration
==============================================================
Based on "MobileNetV4 - Universal Models for the Mobile Ecosystem" (2024)

Key innovations over EfficientNetV2/V4:

1. Universal Inverted Bottleneck (UIB): Unified block that can represent
   MBConv, FusedMBConv, ConvNext-like, and Extra-DW blocks through optional
   depthwise convolutions at start and middle positions
2. Extra-DW Pattern: DW conv BEFORE expansion captures spatial info at
   input resolution (low channel count = very cheap), then standard DW
   after expansion. Best of both worlds: spatial + channel processing
3. Mobile MQA (Multi-Query Attention): Efficient attention with single
   shared Key/Value head across all query heads — 4-8x cheaper than MHSA
4. Mixed kernel sizes within UIB: 3×3 start-DW + 5×5 mid-DW for
   multi-scale receptive fields at minimal extra cost
5. LayerScale + cosine stochastic depth for training stability

Architecture stages for YOLO compatibility:
- MNV4Conv: Fused Inverted Bottleneck stages (P1-P3) — efficient early features
- MNV4UIB: Extra-DW Universal Inverted Bottleneck (P4-P5) — rich spatial+channel
- MNV4Hybrid: UIB + Mobile MQA for deepest stage (P5) — global reasoning
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
# DROP PATH (STOCHASTIC DEPTH) — Cosine schedule variant
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


# ────────────────────────────────────────────────────────────────────────
# LAYER SCALE — Per-channel learnable scaling for training stability
# ────────────────────────────────────────────────────────────────────────
class MNV4LayerScale(nn.Module):
    """
    LayerScale from CaiT/ConvNeXt — learnable per-channel scaling.

    Prevents early-stage gradient explosion by starting with small scale.
    Args:
        dim: Number of channels
        init_value: Initial scale value (default 1e-4)
    """
    def __init__(self, dim, init_value=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma.view(1, -1, 1, 1)


# ────────────────────────────────────────────────────────────────────────
# SQUEEZE-AND-EXCITATION for MobileNetV4
# ────────────────────────────────────────────────────────────────────────
class MNV4SE(nn.Module):
    """
    Squeeze-and-Excitation block for MobileNetV4.

    Uses global average pooling with compact FC layers for channel attention.
    Applied selectively in UIB blocks at deeper stages.

    Args:
        c_in: Input channels (for SE ratio calculation)
        c_expand: Expanded channels (actual input to SE)
        se_ratio: Reduction ratio (default 0.25)
    """
    def __init__(self, c_in, c_expand, se_ratio=0.25):
        super().__init__()
        c_reduced = max(1, int(c_in * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c_expand, c_reduced, 1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(c_reduced, c_expand, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        scale = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * scale


# ────────────────────────────────────────────────────────────────────────
# FUSED INVERTED BOTTLENECK BLOCK (for early stages)
# ────────────────────────────────────────────────────────────────────────
class FusedIBBlock(nn.Module):
    """
    Fused Inverted Bottleneck Block for MobileNetV4.

    Used in early stages (P1-P3) where fused 3×3 convolution is more
    efficient than separate expansion + depthwise + projection.

    Architecture:
    - expand == 1: Simple 3×3 Conv → BN → SiLU
    - expand > 1: 3×3 Conv (fused expand) → BN → SiLU → 1×1 Conv (project) → BN

    Args:
        c1: Input channels
        c2: Output channels
        s: Stride (1 or 2)
        expand: Expansion ratio
        drop_prob: Drop path probability
    """
    def __init__(self, c1, c2, s=1, expand=4, drop_prob=0.0):
        super().__init__()
        hidden_c = int(c1 * expand)
        self.use_res = (s == 1 and c1 == c2)

        if expand == 1:
            self.block = nn.Sequential(
                nn.Conv2d(c1, c2, 3, s, autopad(3), bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                # Fused 3×3 expansion
                nn.Conv2d(c1, hidden_c, 3, s, autopad(3), bias=False),
                nn.BatchNorm2d(hidden_c),
                nn.SiLU(inplace=True),
                # 1×1 projection
                nn.Conv2d(hidden_c, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2),
            )

        self.ls = MNV4LayerScale(c2, init_value=1e-4) if self.use_res else nn.Identity()
        self.drop_path = DropPath(drop_prob) if (self.use_res and drop_prob > 0) else nn.Identity()

    def forward(self, x):
        if self.use_res:
            return x + self.drop_path(self.ls(self.block(x)))
        return self.block(x)


# ────────────────────────────────────────────────────────────────────────
# UNIVERSAL INVERTED BOTTLENECK (UIB) BLOCK — Core MobileNetV4 innovation
# ────────────────────────────────────────────────────────────────────────
class UIBBlock(nn.Module):
    """
    Universal Inverted Bottleneck Block from MobileNetV4.

    The key innovation: two OPTIONAL depthwise convolutions at different positions
    create a unified block that subsumes MBConv, FusedMBConv, ConvNext, and Extra-DW.

    Block architecture:
    [Optional DW-start] → 1×1 Expand → [Optional DW-mid] → SE → 1×1 Project

    | dw_start_k | dw_mid_k | Block Type                    |
    |------------|----------|-------------------------------|
    | 0          | 3        | Standard MBConv (IB)          |
    | 0          | 5        | MBConv with larger kernel      |
    | 3          | 0        | ConvNext-like                  |
    | 3          | 3        | Extra-DW (3×3 + 3×3)          |
    | 3          | 5        | Extra-DW mixed (3×3 + 5×5) ★  |
    | 5          | 5        | Extra-DW large kernels         |

    ★ = Default for P4/P5 stages: captures multi-scale spatial features

    The Extra-DW pattern is the key MobileNetV4 advantage:
    - DW-start processes spatial features at INPUT channel count (very cheap!)
    - After 1×1 expansion to higher channel count, DW-mid refines features
    - Two spatial processing steps at different channel dimensions = richer features
    - Total cost is only marginally higher than standard MBConv

    Args:
        c1: Input channels
        c2: Output channels
        s: Stride (1 or 2)
        expand: Expansion ratio
        dw_start_k: Kernel size for start depthwise conv (0 = skip)
        dw_mid_k: Kernel size for middle depthwise conv (0 = skip)
        se_ratio: SE ratio (0 = no SE)
        drop_prob: Drop path probability
    """
    def __init__(self, c1, c2, s=1, expand=4, dw_start_k=3, dw_mid_k=5,
                 se_ratio=0.25, drop_prob=0.0):
        super().__init__()
        hidden_c = int(c1 * expand)
        self.use_res = (s == 1 and c1 == c2)

        # Determine stride placement:
        # - Extra-DW (both DW present): stride at DW-start
        # - MBConv (only DW-mid): stride at DW-mid
        # - ConvNext (only DW-start): stride at DW-start
        stride_start = s if (dw_start_k > 0) else 1
        stride_mid = s if (dw_start_k == 0 and dw_mid_k > 0) else 1

        layers = []

        # 1. Optional DW-start: spatial processing at input channels (CHEAP!)
        if dw_start_k > 0:
            layers.extend([
                nn.Conv2d(c1, c1, dw_start_k, stride_start,
                          autopad(dw_start_k), groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.SiLU(inplace=True),
            ])

        # 2. 1×1 Expansion (pointwise)
        layers.extend([
            nn.Conv2d(c1, hidden_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.SiLU(inplace=True),
        ])

        # 3. Optional DW-mid: spatial processing at expanded channels
        if dw_mid_k > 0:
            layers.extend([
                nn.Conv2d(hidden_c, hidden_c, dw_mid_k, stride_mid,
                          autopad(dw_mid_k), groups=hidden_c, bias=False),
                nn.BatchNorm2d(hidden_c),
                nn.SiLU(inplace=True),
            ])
        elif dw_start_k == 0 and s > 1:
            # Neither DW present but need stride
            layers.append(nn.AvgPool2d(s, s))

        # 4. SE attention (optional)
        if se_ratio > 0:
            layers.append(MNV4SE(c_in=c1, c_expand=hidden_c, se_ratio=se_ratio))

        # 5. 1×1 Projection (pointwise)
        layers.extend([
            nn.Conv2d(hidden_c, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
        ])

        self.block = nn.Sequential(*layers)

        # LayerScale + DropPath on residual
        self.ls = MNV4LayerScale(c2, init_value=1e-4) if self.use_res else nn.Identity()
        self.drop_path = DropPath(drop_prob) if (self.use_res and drop_prob > 0) else nn.Identity()

    def forward(self, x):
        if self.use_res:
            return x + self.drop_path(self.ls(self.block(x)))
        return self.block(x)


# ────────────────────────────────────────────────────────────────────────
# MOBILE MULTI-QUERY ATTENTION (MQA) — Efficient global attention
# ────────────────────────────────────────────────────────────────────────
class MobileMQA(nn.Module):
    """
    Mobile Multi-Query Attention from MobileNetV4.

    Key innovation over standard MHSA:
    - Multiple query heads but SINGLE shared key/value head
    - 4-8x cheaper than standard MHSA in memory and compute
    - Captures global spatial relationships efficiently
    - Spatial downsampling option for very large feature maps

    Why MQA works: In practice, the key/value information is highly
    redundant across heads. Sharing K/V loses minimal accuracy while
    dramatically reducing parameters and memory bandwidth.

    Args:
        dim: Number of input/output channels
        num_heads: Number of query heads (default 4)
        spatial_ds: Spatial downsampling factor for K/V (default 1 = no downsampling)
    """
    def __init__(self, dim, num_heads=4, spatial_ds=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.spatial_ds = spatial_ds

        # Layer norm before attention
        self.norm = nn.GroupNorm(1, dim)  # Instance norm equivalent, works for conv features

        # Multi-Query: Q has all heads, K and V have single head
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)             # All heads
        self.k_proj = nn.Conv2d(dim, self.head_dim, 1, bias=False)   # Single head
        self.v_proj = nn.Conv2d(dim, self.head_dim, 1, bias=False)   # Single head
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)

        # Spatial downsampling for K/V efficiency
        if spatial_ds > 1:
            self.kv_downsample = nn.AvgPool2d(spatial_ds, spatial_ds)
        else:
            self.kv_downsample = nn.Identity()

        # LayerScale for residual
        self.ls = MNV4LayerScale(dim, init_value=1e-4)

    def forward(self, x):
        B, C, H, W = x.shape

        # Normalize
        x_norm = self.norm(x)

        # Queries: B, num_heads, HW, head_dim
        q = self.q_proj(x_norm)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # Keys/Values: single head, spatially downsampled
        x_ds = self.kv_downsample(x_norm)
        _, _, Hd, Wd = x_ds.shape

        k = self.k_proj(x_ds)                    # B, head_dim, Hd, Wd
        k = k.reshape(B, 1, self.head_dim, Hd * Wd).permute(0, 1, 3, 2)  # B, 1, HdWd, head_dim

        v = self.v_proj(x_ds)                    # B, head_dim, Hd, Wd
        v = v.reshape(B, 1, self.head_dim, Hd * Wd).permute(0, 1, 3, 2)  # B, 1, HdWd, head_dim

        # Attention: Q @ K^T → softmax → @ V
        # K/V broadcast across all query heads (multi-query pattern)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, num_heads, HW, HdWd
        attn = attn.softmax(dim=-1)
        out = attn @ v  # B, num_heads, HW, head_dim

        # Reshape back to spatial
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.out_proj(out)

        # Residual with LayerScale
        return x + self.ls(out)


# ════════════════════════════════════════════════════════════════════════
# STAGE CONTAINERS — YOLO-compatible modules
# ════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────
# MNV4Conv — Fused Inverted Bottleneck Stage (Early stages P1-P3)
# ────────────────────────────────────────────────────────────────────────
class MNV4Conv(nn.Module):
    """
    Stack of Fused Inverted Bottleneck blocks with cosine stochastic depth.

    Used for early stages (P1-P3) where fused 3×3 convolutions are more
    hardware-efficient than separate depthwise + pointwise operations.

    Cosine schedule: dp(i) = max_dp * (1 - cos(π·i/(n-1))) / 2
    → Gentle start, stronger regularization in later blocks

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of blocks (controlled by depth multiplier)
        s: Stride for first block (1 or 2)
        expand: Expansion ratio
        drop_prob: Maximum drop path probability
    """
    def __init__(self, c1, c2, n=1, s=1, expand=4, drop_prob=0.0):
        super().__init__()
        blocks = []
        for i in range(n):
            dp = drop_prob * (1 - math.cos(math.pi * i / max(n - 1, 1))) / 2 if n > 1 else drop_prob
            if i == 0:
                blocks.append(FusedIBBlock(c1, c2, s=s, expand=expand, drop_prob=dp))
            else:
                blocks.append(FusedIBBlock(c2, c2, s=1, expand=expand, drop_prob=dp))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ────────────────────────────────────────────────────────────────────────
# MNV4UIB — Universal Inverted Bottleneck Stage (Later stages P4-P5)
# ────────────────────────────────────────────────────────────────────────
class MNV4UIB(nn.Module):
    """
    Stack of Universal Inverted Bottleneck (UIB) blocks with Extra-DW pattern.

    The Extra-DW pattern (dw_start=3×3, dw_mid=5×5) is the key MobileNetV4
    innovation for deeper stages:
    - DW-start at input channels: Cheap spatial processing before expansion
    - 1×1 expansion to higher channels
    - DW-mid at expanded channels: Rich spatial processing with larger kernel
    - Optional SE attention for channel recalibration (off by default for lightweight)
    - 1×1 projection back to output channels

    This captures spatial features at TWO different channel dimensions,
    providing richer representations than standard MBConv at minimal extra cost.

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of blocks
        s: Stride for first block
        expand: Expansion ratio
        se_ratio: SE ratio (0 = no SE, default 0 for lightweight)
        drop_prob: Maximum drop path probability
    """
    def __init__(self, c1, c2, n=1, s=1, expand=4, se_ratio=0.0, drop_prob=0.0):
        super().__init__()
        blocks = []
        for i in range(n):
            dp = drop_prob * (1 - math.cos(math.pi * i / max(n - 1, 1))) / 2 if n > 1 else drop_prob

            # Alternate between different UIB configurations within a stage
            # for richer feature diversity (inspired by MobileNetV4 NAS results)
            if i % 2 == 0:
                # Extra-DW with mixed kernels: 3×3 start + 5×5 mid
                dw_start_k, dw_mid_k = 3, 5
            else:
                # Extra-DW with uniform kernels: 3×3 + 3×3
                dw_start_k, dw_mid_k = 3, 3

            if i == 0:
                blocks.append(UIBBlock(c1, c2, s=s, expand=expand,
                                       dw_start_k=dw_start_k, dw_mid_k=dw_mid_k,
                                       se_ratio=se_ratio, drop_prob=dp))
            else:
                blocks.append(UIBBlock(c2, c2, s=1, expand=expand,
                                       dw_start_k=dw_start_k, dw_mid_k=dw_mid_k,
                                       se_ratio=se_ratio, drop_prob=dp))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ────────────────────────────────────────────────────────────────────────
# MNV4Hybrid — UIB + Mobile MQA Hybrid Stage (Deepest stage P5)
# ────────────────────────────────────────────────────────────────────────
class MNV4Hybrid(nn.Module):
    """
    Hybrid stage combining UIB blocks with Mobile MQA attention.

    The hybrid design from MobileNetV4:
    - UIB blocks handle local spatial features (convolution-based)
    - MQA blocks add global context reasoning (attention-based)
    - Alternating pattern: UIB → MQA → UIB → MQA → ...
    - MQA uses shared K/V for extreme efficiency

    This is the most expressive stage type, used at the deepest level (P5)
    where global context is most beneficial for detection accuracy.

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of UIB blocks (MQA blocks = n//2, at least 1)
        s: Stride for first block
        expand: Expansion ratio
        se_ratio: SE ratio (0 = no SE, default 0 for lightweight)
        drop_prob: Maximum drop path probability
    """
    def __init__(self, c1, c2, n=1, s=1, expand=4, se_ratio=0.0, drop_prob=0.0):
        super().__init__()
        layers = []

        for i in range(n):
            dp = drop_prob * (1 - math.cos(math.pi * i / max(n - 1, 1))) / 2 if n > 1 else drop_prob

            # UIB block
            if i == 0:
                layers.append(UIBBlock(c1, c2, s=s, expand=expand,
                                       dw_start_k=3, dw_mid_k=5,
                                       se_ratio=se_ratio, drop_prob=dp))
            else:
                layers.append(UIBBlock(c2, c2, s=1, expand=expand,
                                       dw_start_k=3, dw_mid_k=5,
                                       se_ratio=se_ratio, drop_prob=dp))

            # Add MQA after every UIB block (or every other for n>2)
            if n <= 2 or i % 2 == 0:
                # Determine num_heads based on channel count
                num_heads = max(2, min(8, c2 // 32))
                # Ensure c2 is divisible by num_heads
                while c2 % num_heads != 0 and num_heads > 1:
                    num_heads -= 1
                layers.append(MobileMQA(c2, num_heads=num_heads, spatial_ds=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ────────────────────────────────────────────────────────────────────────
# FEATURE ALIGN — Backbone → Neck channel bridging
# ────────────────────────────────────────────────────────────────────────
class FeatureAlignMNV4(nn.Module):
    """
    Feature alignment for backbone → neck channel calibration.

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
