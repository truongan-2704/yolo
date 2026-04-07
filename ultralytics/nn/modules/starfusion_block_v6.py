# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
C3k2_DCNF_V6: Recall-Precision Balanced StarFusion Block — Version 6
=====================================================================
New version — does NOT overwrite any existing starfusion_block files.

Root cause of V5's low Recall (high Precision but misses objects):

1. **Star multiplication `L × D` is an AND-gate**: features only survive if
   BOTH branches agree → weak/uncertain detections from minority classes
   are suppressed → low Recall for underrepresented classes.

2. **SE Gate (reduction=4) double-filters after star**: combined with star's
   multiplicative suppression, this creates excessive feature filtering
   that eliminates borderline-positive detections.

3. **LayerScale γ=1e-4 too conservative**: DCNF branch contributes almost
   nothing early in training → model learns to be overly selective.

V6 Fixes:

1. **Soft Star Fusion**: `L * D + β * (L + D) / 2`
   - Star (AND-gate): captures feature interactions (high Precision)
   - Additive (OR-gate): rescues features if EITHER branch detects (high Recall)
   - Learnable β starts at 0.5, balancing AND/OR automatically
   - Net effect: weak features from minority classes are preserved

2. **Partial SE Attention (mix ratio)**: `α * SE(x) + (1-α) * x`
   - Instead of fully gating, blend attended + unattended features
   - α = learnable, starts at 0.5 → 50% features pass ungated
   - Prevents SE from completely suppressing minority class channels

3. **Larger LayerScale init (1e-2 vs 1e-4)**: DCNF branch contributes
   meaningfully from the start → richer feature learning, less conservative

4. **3rd lightweight branch (1×1 Conv)**: adds channel mixing without
   spatial overhead → provides richer features for star operation,
   especially helping underrepresented classes

Architecture:
    Input → Conv1×1(reduce c1→c_)
    → GroupConv 3×3 g=c_//16 (local, RF=3×3)
    → GroupConv 3×3 d=2 g=c_//16 (dilated, RF=7×7)
    → Conv 1×1 (channel mix — lightweight 3rd branch)
    → ★ Soft Star: L×D + β×(L+D)/2 + sigmoid(α)×h
    → BatchNorm (stabilizer)
    → Partial SE (blended gating)
    → Conv1×1(expand c_→c2)
    → LayerScale γ=1e-2 + Residual → Output
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class PartialSEGate(nn.Module):
    """
    Partial SE Gate — blends attended + unattended features.

    Instead of `x * gate(x)` which can fully zero-out channels,
    uses `mix * SE(x) + (1-mix) * x` where mix is learnable.

    This prevents the SE gate from completely suppressing channels
    that carry weak but important features (minority classes).

    At initialization, mix=0.5 → 50% of features pass ungated,
    ensuring weak detections aren't eliminated during early training.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        c_reduced = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, c_reduced, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_reduced, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        # Learnable mix ratio: how much SE gating to apply
        # Initialize at 0.0 → sigmoid(0.0) = 0.5 → 50/50 blend
        self.mix_weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        mix = torch.sigmoid(self.mix_weight)
        return mix * (x * self.gate(x)) + (1.0 - mix) * x


class StarFusionBottleneck_V6(nn.Module):
    """
    Recall-Precision Balanced StarFusion Bottleneck.

    Architecture: reduce → 3 branches → ★ soft star → partial SE → expand → LayerScale

    Key design principles:
    1. Soft Star: L*D + β*(L+D)/2 → AND-gate + OR-gate balance
    2. Partial SE: blended gating preserves weak features
    3. LayerScale γ=1e-2: stronger initial contribution
    4. 3rd lightweight branch: 1×1 Conv for channel mixing
    5. Enhanced gradient highway: sigmoid(α) * h
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # Ensure groups is at least 1 and divides c_ evenly
        self.groups = max(1, c_ // 16)
        while c_ % self.groups != 0 and self.groups > 1:
            self.groups -= 1

        # Channel reduction: c1 → c_
        self.cv_reduce = Conv(c1, c_, 1, 1)

        # === Branch L (Local): GroupConv 3×3, RF = 3×3 ===
        self.branch_local = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=3, padding=1,
                      groups=self.groups, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # === Branch D (Dilated): GroupConv 3×3 dilation=2, RF = 7×7 ===
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=3, padding=2, dilation=2,
                      groups=self.groups, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # === Branch C (Channel Mix): Conv 1×1 — lightweight 3rd branch ===
        # Provides cross-channel information without spatial overhead
        # Helps minority class features interact with majority class channels
        self.branch_channel = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )

        # === Soft Star Fusion ===
        # β controls the balance between AND-gate (star) and OR-gate (additive)
        # Initialize at 0.0 → sigmoid(0.0) = 0.5 → balanced start
        self.star_beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # α controls the gradient highway strength
        self.star_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # === Star Stabilizer ===
        self.star_bn = nn.BatchNorm2d(c_)

        # === Partial SE Gate ===
        self.se_gate = PartialSEGate(c_, reduction=4)

        # === Channel expansion: c_ → c2 ===
        self.cv_expand = Conv(c_, c2, 1, 1)

        # === LayerScale for stable deep training ===
        # V6: γ=1e-2 (100× larger than V5's 1e-4) for stronger initial contribution
        self.add = shortcut and c1 == c2
        if self.add:
            self.gamma = nn.Parameter(1e-2 * torch.ones((1, c2, 1, 1), dtype=torch.float32))

    def forward(self, x):
        """Forward: reduce → branches → soft star → partial SE → expand → LayerScale."""
        identity = x
        h = self.cv_reduce(x)

        # 1. Three branches
        b_local = self.branch_local(h)
        b_dilated = self.branch_dilated(h)
        b_channel = self.branch_channel(h)

        # 2. ★ Soft Star Fusion
        #    AND-gate: L * D * C captures triple feature interactions (Precision)
        #    OR-gate: (L + D + C) / 3 rescues features if ANY branch detects (Recall)
        #    Gradient highway: α * h ensures gradient flow
        beta = torch.sigmoid(self.star_beta)
        alpha = torch.sigmoid(self.star_alpha)

        star_and = b_local * b_dilated * b_channel           # AND-gate: all must agree
        star_or = (b_local + b_dilated + b_channel) / 3.0    # OR-gate: any can contribute

        star = (1.0 - beta) * star_and + beta * star_or + alpha * h

        # 3. Star stabilizer
        star = self.star_bn(star)

        # 4. Partial SE Gate (blended — preserves weak features)
        star = self.se_gate(star)

        # 5. Expand & LayerScale residual
        out = self.cv_expand(star)
        if self.add:
            return self.gamma * out + identity
        return out


class C3k2_DCNF_V6(nn.Module):
    """
    C3k2_DCNF_V6: Recall-Precision Balanced Dilated Context-Star Fusion.

    Drop-in replacement for C3k2 — uses the same C2f split-concat architecture
    with StarFusionBottleneck_V6 for balanced Precision AND Recall.

    Why V6 achieves better Recall than V5 while maintaining Precision:
    1. Soft Star: AND + OR gate balance → weak features preserved
    2. Partial SE: blended gating → minority class channels not zeroed
    3. LayerScale γ=1e-2: stronger contribution from start
    4. 3rd branch (1×1): channel mixing aids underrepresented classes
    5. Enhanced gradient highway: stable training for all classes

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of StarFusionBottleneck_V6 repeats. Default 1.
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

        # Core: n × StarFusionBottleneck_V6
        self.m = nn.ModuleList(
            StarFusionBottleneck_V6(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward: split → n × StarFusionV6 → concat → merge."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
