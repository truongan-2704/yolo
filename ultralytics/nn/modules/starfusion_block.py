# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
C3k2_DCNF: Dilated Context-Star Fusion Block — Version 1 (Optimized)
=====================================================================
Optimizations applied:
1. Fixed duplicate StarFusionBottleneck class definition (was silently overwriting)
2. Inplace SiLU activations to reduce peak memory usage
3. Simplified ECA gate with cleaner tensor reshaping
4. Added star_bn for post-multiplication stability
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class StarFusionBottleneck(nn.Module):
    """
    Optimized StarFusion Bottleneck: Multi-Scale Dilated Asymmetric Context + Star Operation.
    - BatchNorm post-star operation to prevent gradient explosion.
    - ECA (Efficient Channel Attention) Conv1D gate for lightweight gating.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels (c//2)

        # Channel reduction: c1 → c_
        self.cv_reduce = Conv(c1, c_, 1, 1)

        # === Multi-Scale Dilated Asymmetric Context (MDAC) ===
        self.branch_local = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=3, padding=1, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        self.branch_dilated = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=3, padding=2, dilation=2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        self.branch_asym = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=(1, 5), padding=(0, 2), groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_, c_, kernel_size=(5, 1), padding=(2, 0), groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # === Star Operation Stabilizer ===
        self.star_bn = nn.BatchNorm2d(c_)

        # === Efficient Channel Attention (ECA) Gate ===
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.gate_act = nn.Sigmoid()

        # Channel expansion: c_ → c2
        self.cv_expand = Conv(c_, c2, 1, 1)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        h = self.cv_reduce(x)

        # 1. MDAC branches
        b_local = self.branch_local(h)
        b_dilated = self.branch_dilated(h)
        b_asym = self.branch_asym(h)

        # 2. ★ Star Operation
        star = b_local * b_dilated * b_asym
        star = self.star_bn(star)

        # 3. ECA Context-Aware Gating (optimized reshape)
        b, c = star.shape[:2]
        y = self.gate_pool(star).view(b, 1, c)  # [B, 1, C]
        y = self.gate_conv(y)                     # [B, 1, C]
        gate = self.gate_act(y).view(b, c, 1, 1)  # [B, C, 1, 1]
        star = star * gate

        # 4. Expand & Residual
        out = self.cv_expand(star)
        return out + identity if self.add else out


class C3k2_DCNF(nn.Module):
    """
    C3k2_DCNF: Dilated Context-Star Fusion — drop-in replacement for C3k2.

    Inherits the C2f split-concat architecture but replaces Bottleneck modules
    with StarFusionBottleneck for reduced params/FLOPs and richer multi-scale
    feature representation via star operation.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of StarFusionBottleneck repeats. Default 1.
        c3k (bool): Unused, kept for API compatibility with C3k2. Default False.
        e (float): Expansion ratio. Default 0.5.
        g (int): Groups. Default 1.
        shortcut (bool): Whether to use residual connections in bottlenecks. Default True.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initialize C3k2_DCNF with split-concat architecture and StarFusion bottlenecks."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels per split

        # Entry conv: split input into 2 × self.c channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Exit conv: merge (2 + n) × self.c channels back to c2
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Core: n × StarFusionBottleneck
        self.m = nn.ModuleList(
            StarFusionBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C3k2_DCNF: split → n × StarFusion → concat → merge."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
