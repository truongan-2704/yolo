# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
C3k2_DCNF_V1Plus: Improved StarFusion Block — Version 1+
=========================================================
Evolutionary improvement over V1 (the best-performing version).

Three targeted changes (all novel, not tried in V2/V3):
1. Pairwise Star Fusion: sum of pairwise products instead of triple product
   → better gradient flow, richer 2nd-order interactions
2. Star + Input Residual: adds pre-branch features back after star
   → prevents dead channels when star products approach zero
3. SE Bottleneck Gate: proper SE with reduction ratio r=4
   → fewer params than V1's single-layer SE, more expressive than V2's ECA

Usage:
    Drop-in replacement for C3k2 in any YOLO config YAML.
    C3k2_DCNF_V1Plus(c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True)
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class SpatialAttention(nn.Module):
    """Spatial Attention Module (CS-Gate component)."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial Attention with avg/max pooling and conv layer."""
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention by combining mean and max pooling across channels."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class StarFusionBottleneck_V1Plus(nn.Module):
    """
    Improved StarFusion Bottleneck (V1++) with 3 novel enhancements:
    1. Dynamic Pairwise Fusion: Learnable softmax weighting for RF interactions.
    2. CS-Gate (Channel-Spatial Gate): Dual attention to preserve spatial details.
    3. LayerScale: Scaled residual connections for deeper network stability.

    Architecture:
        Input → Conv1×1 (reduce) → 3 DWConv branches
        → ★ Dynamic Pairwise Star Fusion
        → + Input Residual
        → ★ CS-Gate (Channel + Spatial Attention)
        → Conv1×1 (expand) → ★ LayerScale + Residual → Output
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initialize StarFusionBottleneck_V1Plus with CS-Gate, Dynamic Fusion, and LayerScale."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels (c//2)

        # Channel reduction: c1 → c_
        self.cv_reduce = Conv(c1, c_, 1, 1)

        # === Multi-Scale Dilated Asymmetric Context (MDAC) ===

        # Branch L (Local): DWConv 3×3, RF = 3×3
        self.branch_local = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=3, padding=1, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
        )

        # Branch D (Dilated): DWConv 3×3 dilation=2, RF = 7×7
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=3, padding=2, dilation=2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
        )

        # Branch A (Asymmetric): DWConv 1×5 → DWConv 5×1, RF = cross-shaped 5×5
        self.branch_asym = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=(1, 5), padding=(0, 2), groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
            nn.Conv2d(c_, c_, kernel_size=(5, 1), padding=(2, 0), groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
        )

        # === NEW 1: Learnable Pairwise Weights (Dynamic Star Fusion) ===
        # Parameter to dynamically weigh the 3 pairs: (L*D), (L*A), (D*A)
        self.pw_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))

        # Learnable residual weight for star + input fusion
        self.star_residual_weight = nn.Parameter(torch.tensor(0.1))

        # === NEW 2: CS-Gate (Channel-Spatial Gate) ===
        r = 4  # reduction ratio
        c_squeeze = max(c_ // r, 8)  # ensure minimum 8 channels
        
        # Channel Gate (SE)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_, c_squeeze, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(c_squeeze, c_, 1, bias=True),
            nn.Sigmoid(),
        )
        
        # Spatial Gate
        self.spatial_gate = SpatialAttention(kernel_size=7)

        # Channel expansion: c_ → c2
        self.cv_expand = Conv(c_, c2, 1, 1)

        # === NEW 3: LayerScale for robust Residual Connection ===
        self.add = shortcut and c1 == c2
        if self.add:
            # Initialize LayerScale parameter with a small value (ConvNeXt style)
            self.gamma = nn.Parameter(1e-4 * torch.ones((1, c2, 1, 1), dtype=torch.float32))
        else:
            self.gamma = None

    def forward(self, x):
        """Forward pass: reduce → MDAC → dynamic pairwise star → CS-Gate → expand → LayerScale."""
        identity = x

        # 1. Channel reduction
        h = self.cv_reduce(x)

        # 2. Multi-Scale Dilated Asymmetric Context (MDAC)
        b_local = self.branch_local(h)
        b_dilated = self.branch_dilated(h)
        b_asym = self.branch_asym(h)

        # 3. ★ NEW: Dynamic Pairwise Star Fusion
        w = torch.softmax(self.pw_weights, dim=0)
        star = (w[0] * (b_local * b_dilated) + 
                w[1] * (b_local * b_asym) + 
                w[2] * (b_dilated * b_asym))

        # 4. Star + Input Residual (prevent dead channels)
        star = star + self.star_residual_weight * h

        # 5. ★ NEW: CS-Gate (Channel-Spatial Gating)
        c_gate = self.channel_gate(star)
        s_gate = self.spatial_gate(star)
        
        # Combine gates (Broadcasting applies correctly)
        star = star * c_gate * s_gate

        # 6. Channel expansion
        out = self.cv_expand(star)

        # 7. ★ NEW: Residual connection with LayerScale
        return self.gamma * out + identity if self.add else out


class C3k2_DCNF_V1Plus(nn.Module):
    """
    C3k2_DCNF_V1Plus: Improved Dilated Context-Star Fusion — drop-in replacement for C3k2.

    Same C2f split-concat architecture as V1, using StarFusionBottleneck_V1Plus.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of StarFusionBottleneck_V1Plus repeats. Default 1.
        c3k (bool): Unused, kept for API compatibility with C3k2. Default False.
        e (float): Expansion ratio. Default 0.5.
        g (int): Groups. Default 1.
        shortcut (bool): Whether to use residual connections in bottlenecks. Default True.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initialize C3k2_DCNF_V1Plus with split-concat architecture and improved bottlenecks."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels per split

        # Entry conv: split input into 2 × self.c channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Exit conv: merge (2 + n) × self.c channels back to c2
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Core: n × StarFusionBottleneck_V1Plus
        self.m = nn.ModuleList(
            StarFusionBottleneck_V1Plus(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass: split → n × StarFusionV1Plus → concat → merge."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
