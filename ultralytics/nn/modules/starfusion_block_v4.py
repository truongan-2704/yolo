# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
C3k2_DCNF_V4: EfficientNet-Aware StarFusion Block — Version 4
==============================================================
Designed specifically for optimal synergy with EfficientNetV2 backbone.

Key innovations over V1 (best baseline):

1. **Adaptive Multi-Scale Fusion (AMSF)**: Instead of V1's fixed 3-way star
   multiplication (local * dilated * asym), V4 uses pairwise star products
   with learnable softmax weights — prevents gradient vanishing from 3-way
   multiplication while maintaining multi-scale interaction.

2. **Efficient SE-Gating (ESE)**: Replaces V1's ECA (Conv1d) with an
   EfficientNet-compatible SE gate using Hardsigmoid. This creates a
   seamless "attention vocabulary" between backbone (EfficientNetV2 SE)
   and neck (C3k2_DCNF_V4 ESE) — the network learns consistent
   channel relationships end-to-end.

3. **Gradient-Friendly Star Residual**: Adds a learnable-weighted
   shortcut from pre-star features to post-star output, preventing
   dead channels from multiplicative operations.

4. **Proper Initialization**: BatchNorm after star operation + 
   small initial LayerScale gamma for stable early training.

5. **Reduced Complexity**: Cleaner architecture than V1Plus/V2/V3
   while being more effective — no DCN overhead (V3), no parameter-free
   gate complexity (V2), no dual attention overhead (V1Plus).

Architecture:
    Input → Conv1×1(reduce) → 3 DWConv branches (local/dilated/asym)
    → ★ Weighted Pairwise Star Fusion + star_bn
    → + Learnable Star Residual
    → ★ Efficient SE-Gate (Hardsigmoid)
    → Conv1×1(expand) → LayerScale + Residual → Output
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class EfficientSEGate(nn.Module):
    """
    Efficient SE Gate — compatible with EfficientNetV2's attention style.

    Uses the same Hardsigmoid activation as the improved EfficientNetV2 SE block,
    creating a consistent channel attention "vocabulary" throughout the network.
    This means channel importance patterns learned in the backbone are naturally
    continued and refined in the neck, rather than being "re-learned" with a
    different attention mechanism.

    Compared to ECA (V1's Conv1d gate):
    - Better channel interaction modeling (FC layers vs local Conv1d)
    - Matches backbone's attention distribution → smoother gradient flow
    - Hardsigmoid is ~30% faster than Sigmoid
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        c_reduced = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, c_reduced, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_reduced, channels, 1, bias=True),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        return x * self.gate(x)


class StarFusionBottleneck_V4(nn.Module):
    """
    EfficientNet-Aware StarFusion Bottleneck.

    Architecture: reduce → MDAC → pairwise star → star_bn → star_residual → ESE → expand → LayerScale

    Key design decisions:
    1. Pairwise star (not 3-way): L*D + L*A + D*A prevents gradient vanishing
    2. ESE gate matches backbone's SE style → consistent attention vocabulary  
    3. LayerScale with small init → stable deep network training
    4. Star residual prevents dead channels from multiplication
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # Channel reduction: c1 → c_
        self.cv_reduce = Conv(c1, c_, 1, 1)

        # === Multi-Scale Dilated Asymmetric Context (MDAC) ===
        # Branch L (Local): DWConv 3×3, RF = 3×3
        self.branch_local = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=3, padding=1, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # Branch D (Dilated): DWConv 3×3 dilation=2, RF = 7×7
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=3, padding=2, dilation=2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # Branch A (Asymmetric): DWConv 1×5 → DWConv 5×1, RF = cross-shaped 5×5
        self.branch_asym = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=(1, 5), padding=(0, 2), groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_, c_, kernel_size=(5, 1), padding=(2, 0), groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # === Pairwise Star Fusion Weights ===
        # 3 learnable weights for: L*D, L*A, D*A
        self.pw_weights = nn.Parameter(torch.ones(3, dtype=torch.float32))

        # === Star Operation Stabilizer ===
        self.star_bn = nn.BatchNorm2d(c_)

        # === Star Residual (prevents dead channels) ===
        self.star_residual_weight = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # === Efficient SE-Gate (EfficientNet-compatible) ===
        self.ese_gate = EfficientSEGate(c_, reduction=4)

        # === Channel expansion: c_ → c2 ===
        self.cv_expand = Conv(c_, c2, 1, 1)

        # === LayerScale for stable deep training ===
        self.add = shortcut and c1 == c2
        if self.add:
            self.gamma = nn.Parameter(1e-4 * torch.ones((1, c2, 1, 1), dtype=torch.float32))

    def forward(self, x):
        """Forward: reduce → MDAC → pairwise star → ESE → expand → LayerScale."""
        identity = x
        h = self.cv_reduce(x)

        # 1. MDAC branches
        b_local = self.branch_local(h)
        b_dilated = self.branch_dilated(h)
        b_asym = self.branch_asym(h)

        # 2. ★ Weighted Pairwise Star Fusion
        # Softmax ensures weights sum to 1, stable optimization
        w = torch.softmax(self.pw_weights, dim=0)
        star = (w[0] * (b_local * b_dilated) +
                w[1] * (b_local * b_asym) +
                w[2] * (b_dilated * b_asym))

        # 3. Star stabilizer + residual from pre-star features
        star = self.star_bn(star)
        star = star + self.star_residual_weight * h

        # 4. ESE-Gate (EfficientNet-compatible channel attention)
        star = self.ese_gate(star)

        # 5. Expand & LayerScale residual
        out = self.cv_expand(star)
        if self.add:
            return self.gamma * out + identity
        return out


class C3k2_DCNF_V4(nn.Module):
    """
    C3k2_DCNF_V4: EfficientNet-Aware Dilated Context-Star Fusion.

    Drop-in replacement for C3k2 — uses the same C2f split-concat architecture
    with StarFusionBottleneck_V4 for optimal EfficientNetV2 backbone synergy.

    Why V4 works better with EfficientNetV2:
    1. ESE gate uses same Hardsigmoid as backbone → consistent attention
    2. Pairwise star avoids gradient vanishing from 3-way multiply
    3. LayerScale enables stable training of deeper configurations
    4. Star residual prevents feature collapse in early training

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of StarFusionBottleneck_V4 repeats. Default 1.
        c3k (bool): Unused, kept for API compatibility with C3k2. Default False.
        e (float): Expansion ratio. Default 0.5.
        g (int): Groups. Default 1.
        shortcut (bool): Whether to use residual connections. Default True.
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels per split

        # Entry conv: split input into 2 × self.c channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Exit conv: merge (2 + n) × self.c channels back to c2
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Core: n × StarFusionBottleneck_V4
        self.m = nn.ModuleList(
            StarFusionBottleneck_V4(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward: split → n × StarFusionV4 → concat → merge."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
