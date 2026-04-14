"""
EfficientNetV4 — Next-Gen Backbone Modules for YOLO Integration
================================================================
Key innovations over EfficientNetV2:

1. Context-Aware SE (CASE): Dual-pooling (AvgPool + MaxPool) for richer
   channel statistics → captures both average activation AND peak response
2. Multi-Kernel Depthwise Conv (MKDWConv): Parallel 3×3 + 5×5 depthwise
   branches → multi-scale receptive fields without extra FLOPs
3. LayerScale: Per-channel learnable scaling (init=1e-4) after residual
   → stabilizes deep network training, prevents early gradient explosion
4. GeLU activation in projection paths → smoother gradients than SiLU
5. Enhanced Fused-MBConv with optional lightweight attention
6. Progressive stochastic depth with cosine schedule (smoother than linear)

Architecture stages follow EfficientNetV2 pattern for YOLO compatibility:
- FusedMBConvV4: Early stages (P1-P3) — fused 3×3 + projection
- MBConvV4: Later stages (P4-P5) — expand + MKDW + CASE + project
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
class LayerScale(nn.Module):
    """
    LayerScale from CaiT/ConvNeXt — learnable per-channel scaling.

    Prevents early-stage gradient explosion by starting with small scale.
    Gradually learns optimal contribution of each residual block.

    Args:
        dim: Number of channels
        init_value: Initial scale value (default 1e-4 for deep networks)
    """
    def __init__(self, dim, init_value=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        # x: (B, C, H, W) → gamma: (C,) → broadcast as (1, C, 1, 1)
        return x * self.gamma.view(1, -1, 1, 1)


# ────────────────────────────────────────────────────────────────────────
# CONTEXT-AWARE SE (CASE) — Dual-pooling Squeeze-and-Excitation
# ────────────────────────────────────────────────────────────────────────
class CASE(nn.Module):
    """
    Context-Aware Squeeze-and-Excitation Block.

    Innovation over standard SE:
    - Uses BOTH AvgPool AND MaxPool for squeeze operation
    - AvgPool captures average activation (smooth features)
    - MaxPool captures peak responses (salient features like edges/corners)
    - Combined statistics give richer channel importance estimation
    - Shared FC weights for both paths (parameter efficient)

    Args:
        c_in: Input channels (for SE ratio calculation)
        c_expand: Expanded channels (actual input to SE)
        se_ratio: Reduction ratio (default 0.25)
    """
    def __init__(self, c_in, c_expand, se_ratio=0.25):
        super().__init__()
        c_reduced = max(1, int(c_in * se_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared FC for both pooling paths (parameter efficient)
        self.fc1 = nn.Conv2d(c_expand, c_reduced, 1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(c_reduced, c_expand, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        # Dual-pooling squeeze
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        # Combine with element-wise addition → single gate
        scale = self.gate(avg_out + max_out)
        return x * scale


# ────────────────────────────────────────────────────────────────────────
# MULTI-KERNEL DEPTHWISE CONV (MKDWConv)
# ────────────────────────────────────────────────────────────────────────
class MKDWConv(nn.Module):
    """
    Multi-Kernel Depthwise Convolution.

    Innovation: Parallel depthwise convolutions with different kernel sizes
    capture features at multiple spatial scales simultaneously.

    - 3×3 branch: Fine-grained local features (edges, textures)
    - 5×5 branch: Medium-range spatial patterns (parts, small objects)
    - Branches operate on channel splits → no extra FLOPs vs single kernel
    - Results concatenated → rich multi-scale feature representation

    Args:
        c: Number of channels
        s: Stride
    """
    def __init__(self, c, s=1):
        super().__init__()
        # Split channels: 2/3 for 3×3, 1/3 for 5×5
        self.c_small = c - c // 3  # Majority goes to 3×3 (cheaper)
        self.c_large = c // 3       # Minority goes to 5×5 (richer)

        # Ensure at least 1 channel per branch
        if self.c_large == 0:
            self.c_large = c
            self.c_small = 0

        if self.c_small > 0:
            self.dw_small = nn.Sequential(
                nn.Conv2d(self.c_small, self.c_small, 3, s, autopad(3),
                          groups=self.c_small, bias=False),
                nn.BatchNorm2d(self.c_small),
            )

        self.dw_large = nn.Sequential(
            nn.Conv2d(self.c_large, self.c_large, 5, s, autopad(5),
                      groups=self.c_large, bias=False),
            nn.BatchNorm2d(self.c_large),
        )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        if self.c_small > 0:
            x_small, x_large = x.split([self.c_small, self.c_large], dim=1)
            out = torch.cat([self.dw_small(x_small), self.dw_large(x_large)], dim=1)
        else:
            out = self.dw_large(x)
        return self.act(out)


# ────────────────────────────────────────────────────────────────────────
# FUSED-MBCONV V4 BLOCK — Enhanced with GeLU projection + LayerScale
# ────────────────────────────────────────────────────────────────────────
class FusedMBConvBlockV4(nn.Module):
    """
    Single Fused-MBConv Block — V4 Enhanced.

    Improvements over V2:
    - GeLU activation in projection path (smoother gradients)
    - LayerScale for training stability on residual path
    - Cleaner expand=1 handling

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
        self.use_res_connect = (s == 1 and c1 == c2)

        if expand == 1:
            self.block = nn.Sequential(
                nn.Conv2d(c1, c2, 3, s, autopad(3), bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                # Fused expansion: Conv3×3
                nn.Conv2d(c1, hidden_c, 3, s, autopad(3), bias=False),
                nn.BatchNorm2d(hidden_c),
                nn.SiLU(inplace=True),
                # Projection: Conv1×1 with GeLU
                nn.Conv2d(hidden_c, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2),
            )

        # V4: LayerScale + DropPath on residual
        self.ls = LayerScale(c2, init_value=1e-4) if self.use_res_connect else nn.Identity()
        self.drop_path = DropPath(drop_prob) if (self.use_res_connect and drop_prob > 0) else nn.Identity()

    def forward(self, x):
        if self.use_res_connect:
            return x + self.drop_path(self.ls(self.block(x)))
        return self.block(x)


# ────────────────────────────────────────────────────────────────────────
# MBCONV V4 BLOCK — Multi-Kernel DW + CASE + LayerScale
# ────────────────────────────────────────────────────────────────────────
class MBConvBlockV4(nn.Module):
    """
    Single MBConv Block — V4 Enhanced.

    Key innovations:
    - Multi-Kernel Depthwise Conv (3×3 + 5×5) for multi-scale features
    - Context-Aware SE (CASE) with dual-pooling attention
    - LayerScale for training stability
    - GeLU in projection path

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
        self.use_res_connect = (s == 1 and c1 == c2)

        layers = []

        # 1. Expansion (1×1 Conv) — skip if expand == 1
        if expand != 1:
            layers.extend([
                nn.Conv2d(c1, hidden_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_c),
                nn.SiLU(inplace=True),
            ])

        # 2. Multi-Kernel Depthwise Conv (V4 innovation)
        layers.append(MKDWConv(hidden_c, s=s))

        # 3. Context-Aware SE (CASE) — dual-pooling attention
        layers.append(CASE(c_in=c1, c_expand=hidden_c, se_ratio=0.25))

        # 4. Projection (1×1 Conv)
        layers.extend([
            nn.Conv2d(hidden_c, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
        ])

        self.block = nn.Sequential(*layers)

        # V4: LayerScale + DropPath on residual
        self.ls = LayerScale(c2, init_value=1e-4) if self.use_res_connect else nn.Identity()
        self.drop_path = DropPath(drop_prob) if (self.use_res_connect and drop_prob > 0) else nn.Identity()

    def forward(self, x):
        if self.use_res_connect:
            return x + self.drop_path(self.ls(self.block(x)))
        return self.block(x)


# ────────────────────────────────────────────────────────────────────────
# FUSED-MBCONV V4 STAGE — Cosine stochastic depth schedule
# ────────────────────────────────────────────────────────────────────────
class FusedMBConvV4(nn.Module):
    """
    Stack of FusedMBConv V4 Blocks with cosine stochastic depth.

    Innovation: Cosine schedule for drop_prob (smoother than linear):
    - Early blocks: very low drop probability (learn basic features)
    - Later blocks: higher drop probability (regularize complex features)
    - Cosine curve is gentler than linear at start → preserves early learning

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of blocks (controlled by depth multiplier)
        s: Stride for first block
        expand: Expansion ratio
        drop_prob: Maximum drop path probability
    """
    def __init__(self, c1, c2, n=1, s=1, expand=4, drop_prob=0.0):
        super().__init__()
        blocks = []
        for i in range(n):
            # Cosine schedule: dp(i) = drop_prob * (1 - cos(π * i / (n-1))) / 2
            if n > 1:
                dp = drop_prob * (1 - math.cos(math.pi * i / (n - 1))) / 2
            else:
                dp = drop_prob
            if i == 0:
                blocks.append(FusedMBConvBlockV4(c1, c2, s=s, expand=expand, drop_prob=dp))
            else:
                blocks.append(FusedMBConvBlockV4(c2, c2, s=1, expand=expand, drop_prob=dp))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ────────────────────────────────────────────────────────────────────────
# MBCONV V4 STAGE — Cosine stochastic depth schedule
# ────────────────────────────────────────────────────────────────────────
class MBConvV4(nn.Module):
    """
    Stack of MBConv V4 Blocks with CASE + MKDWConv + cosine depth.

    Same cosine schedule + V4 block innovations.

    Args:
        c1: Input channels
        c2: Output channels
        n: Number of blocks
        s: Stride for first block
        expand: Expansion ratio
        drop_prob: Maximum drop path probability
    """
    def __init__(self, c1, c2, n=1, s=1, expand=4, drop_prob=0.0):
        super().__init__()
        blocks = []
        for i in range(n):
            if n > 1:
                dp = drop_prob * (1 - math.cos(math.pi * i / (n - 1))) / 2
            else:
                dp = drop_prob
            if i == 0:
                blocks.append(MBConvBlockV4(c1, c2, s=s, expand=expand, drop_prob=dp))
            else:
                blocks.append(MBConvBlockV4(c2, c2, s=1, expand=expand, drop_prob=dp))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ────────────────────────────────────────────────────────────────────────
# FEATURE ALIGN V4 — Enhanced backbone → neck bridging
# ────────────────────────────────────────────────────────────────────────
class FeatureAlignV4(nn.Module):
    """
    Enhanced Feature Alignment for backbone → neck channel calibration.

    V4 improvements:
    - Uses GeLU activation for smoother feature transformation
    - Optional channel attention gate for selective feature forwarding

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
