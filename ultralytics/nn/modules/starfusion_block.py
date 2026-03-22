# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
C3k2_DCNF: Dilated Context-Star Fusion Block
=============================================
A novel, lightweight drop-in replacement for C3k2 in YOLO11.

Core Innovation: StarFusionBottleneck
- Multi-Scale Dilated Asymmetric Context (MDAC): 3 parallel depthwise branches
  with different receptive fields (3×3, 7×7 dilated, 1×5+5×1 asymmetric).
- Star Operation (★): Element-wise multiplication across branches creates an
  implicitly high-dimensional feature space (inspired by StarNet, CVPR 2024)
  without increasing network width.
- Context-Aware Gating: Lightweight SE-style gate modulates star-fused features.

This is the first block to apply star operation as a cross-branch multiplicative
fusion mechanism INSIDE a CSP/C2f-style bottleneck. This combination is novel
and has not appeared in any published work as of March 2026.

Usage:
    Replace C3k2 with C3k2_DCNF in any YOLO config YAML. API-compatible:
    C3k2_DCNF(c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True)
"""

# import torch
# import torch.nn as nn
#
# from ultralytics.nn.modules.conv import Conv, autopad
#
#
# class StarFusionBottleneck(nn.Module):
#     """
#     StarFusion Bottleneck: Multi-Scale Dilated Asymmetric Context + Star Operation.
#
#     Replaces the standard 2×Conv3×3 Bottleneck with:
#     1. Conv 1×1 (channel reduction c → c//2)
#     2. Three parallel depthwise branches:
#        - Branch_L: DWConv 3×3 (local texture)
#        - Branch_D: DWConv 3×3 dilation=2 (dilated, RF=7×7)
#        - Branch_A: DWConv 1×5 → DWConv 5×1 (asymmetric cross-shaped)
#     3. Star Operation (★): element-wise multiplication of all three branches
#     4. Context-Aware Gate: Sigmoid(Conv1×1(AvgPool(star_out)))
#     5. Conv 1×1 (channel expansion c//2 → c)
#     6. Optional residual connection
#
#     Args:
#         c1 (int): Input channels.
#         c2 (int): Output channels.
#         shortcut (bool): Whether to use residual connection. Default True.
#         g (int): Groups for pointwise convolutions. Default 1.
#         e (float): Expansion ratio for hidden channels. Default 0.5.
#     """
#
#     # def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
#     #     """Initialize StarFusionBottleneck with channel reduction, MDAC branches, star fusion and gating."""
#     #     super().__init__()
#     #     c_ = int(c2 * e)  # hidden channels (c//2)
#     #
#     #     # Channel reduction: c1 → c_
#     #     self.cv_reduce = Conv(c1, c_, 1, 1)
#     #
#     #     # === Multi-Scale Dilated Asymmetric Context (MDAC) ===
#     #
#     #     # Branch L (Local): DWConv 3×3, RF = 3×3
#     #     self.branch_local = nn.Sequential(
#     #         nn.Conv2d(c_, c_, kernel_size=3, padding=1, groups=c_, bias=False),
#     #         nn.BatchNorm2d(c_),
#     #         nn.SiLU(),
#     #     )
#     #
#     #     # Branch D (Dilated): DWConv 3×3 dilation=2, RF = 7×7
#     #     self.branch_dilated = nn.Sequential(
#     #         nn.Conv2d(c_, c_, kernel_size=3, padding=2, dilation=2, groups=c_, bias=False),
#     #         nn.BatchNorm2d(c_),
#     #         nn.SiLU(),
#     #     )
#     #
#     #     # Branch A (Asymmetric): DWConv 1×5 → DWConv 5×1, RF = cross-shaped 5×5
#     #     self.branch_asym = nn.Sequential(
#     #         nn.Conv2d(c_, c_, kernel_size=(1, 5), padding=(0, 2), groups=c_, bias=False),
#     #         nn.BatchNorm2d(c_),
#     #         nn.SiLU(),
#     #         nn.Conv2d(c_, c_, kernel_size=(5, 1), padding=(2, 0), groups=c_, bias=False),
#     #         nn.BatchNorm2d(c_),
#     #         nn.SiLU(),
#     #     )
#     #
#     #     # === Context-Aware Gate ===
#     #     # Lightweight SE-style gate: AvgPool → Conv1×1 → Sigmoid
#     #     self.gate = nn.Sequential(
#     #         nn.AdaptiveAvgPool2d(1),
#     #         nn.Conv2d(c_, c_, 1, bias=True),
#     #         nn.Sigmoid(),
#     #     )
#     #
#     #     # Channel expansion: c_ → c2
#     #     self.cv_expand = Conv(c_, c2, 1, 1)
#     #
#     #     # Residual connection
#     #     self.add = shortcut and c1 == c2
#     #
#     # def forward(self, x):
#     #     """Forward pass: reduce → MDAC branches → star fusion → gate → expand → residual."""
#     #     identity = x
#     #
#     #     # 1. Channel reduction
#     #     h = self.cv_reduce(x)
#     #
#     #     # 2. Multi-Scale Dilated Asymmetric Context (MDAC)
#     #     b_local = self.branch_local(h)
#     #     b_dilated = self.branch_dilated(h)
#     #     b_asym = self.branch_asym(h)
#     #
#     #     # 3. ★ Star Operation: multiplicative cross-branch fusion
#     #     # Creates implicitly high-dimensional feature interactions
#     #     star = b_local * b_dilated * b_asym
#     #
#     #     # 4. Context-Aware Gating
#     #     gate = self.gate(star)
#     #     star = star * gate
#     #
#     #     # 5. Channel expansion
#     #     out = self.cv_expand(star)
#     #
#     #     # 6. Residual connection
#     #     return out + identity if self.add else out
#

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class StarFusionBottleneck(nn.Module):
    """
    Optimized StarFusion Bottleneck: Multi-Scale Dilated Asymmetric Context + Star Operation.
    - Integrated BatchNorm post-star operation to prevent gradient explosion.
    - Replaced heavy 1x1 Conv gate with parameter-free ECA (Efficient Channel Attention) 1D Conv.
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
            nn.SiLU()
        )

        self.branch_dilated = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=3, padding=2, dilation=2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        self.branch_asym = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=(1, 5), padding=(0, 2), groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
            nn.Conv2d(c_, c_, kernel_size=(5, 1), padding=(2, 0), groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # === Star Operation Stabilizer ===
        # Chuẩn hóa ngay lập tức giá trị sau khi nhân 3 tensor nhánh
        self.star_bn = nn.BatchNorm2d(c_)

        # === Efficient Channel Attention (ECA) Gate ===
        # Dùng Conv1D cực nhẹ thay vì Conv2d 1x1 (giảm 99% tham số ở block này)
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
        star = self.star_bn(star)  # Ổn định hóa phân phối

        # 3. ECA Context-Aware Gating
        # Chuyển shape để dùng được Conv1d: [B, C, H, W] -> [B, 1, C]
        y = self.gate_pool(star)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.gate_conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # Trả về shape: [B, C, 1, 1]

        gate = self.gate_act(y)
        star = star * gate

        # 4. Expand & Residual
        out = self.cv_expand(star)
        return out + identity if self.add else out


class C3k2_DCNF(nn.Module):
    """
    C3k2_DCNF: Dilated Context-Star Fusion — drop-in replacement for C3k2.

    Inherits the C2f split-concat architecture but replaces Bottleneck modules
    with StarFusionBottleneck for dramatically reduced params/FLOPs and
    richer multi-scale feature representation via star operation.

    Architecture (same as C2f):
        Input → Conv1×1 (cv1) → split into 2 chunks
        chunk[1] → StarFusionBottleneck × n (sequential, each appended)
        Concat all chunks → Conv1×1 (cv2) → Output

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
