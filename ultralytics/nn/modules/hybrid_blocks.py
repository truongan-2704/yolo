# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Cross-Architecture Hybrid Blocks — 5 Novel Hybrid YOLO Variants
=================================================================

Combines orthogonal decomposition axes from existing custom architectures:

1. NexusPrism   — Direction (Nexus) + Frequency (Prism)
2. PrismEdge    — Frequency (Prism) + Sparsity (Edge)  
3. PhoenixNexus — Scale (Phoenix) + Direction (Nexus)
4. ChimeraPrism — Dilation (Chimera) + Frequency (Prism)
5. SpectraEdge  — Wavelet (Spectra) + Sparsity (Edge)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


# ═════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _channel_shuffle(x, groups):
    """Shuffle channels between groups for cross-group information flow."""
    B, C, H, W = x.shape
    if C % groups != 0:
        return x
    return x.view(B, groups, C // groups, H, W).transpose(1, 2).contiguous().view(B, C, H, W)


# ═════════════════════════════════════════════════════════════════════════════
# 1. YOLO-NEXUS-PRISM — Direction + Frequency Fusion
# ═════════════════════════════════════════════════════════════════════════════

class NexusPrismConv(nn.Module):
    """
    Dual-axis spatial decomposition: Direction (OmniDir) + Frequency (TriFreq).
    
    Splits channels into 2 halves:
    - Half 1: 4-way directional DWConv (iso/hor/ver/dilated) from Nexus
    - Half 2: 3-band frequency DWConv (LF/MF/HF) from Prism
    Then concatenates and shuffles for cross-axis mixing.
    """

    def __init__(self, c, k_strip=5, dilation=2, k_lo=5, k_hi=3):
        super().__init__()
        # Split into direction half and frequency half
        self.c_dir = c // 2
        self.c_freq = c - self.c_dir

        # ── Direction half: 4-way OmniDir ──
        c_d = self.c_dir
        self.c_iso = c_d // 4
        self.c_hor = c_d // 4
        self.c_ver = c_d // 4
        self.c_dil = c_d - self.c_iso - self.c_hor - self.c_ver

        self.dw_iso = nn.Conv2d(self.c_iso, self.c_iso, 3, 1, 1, groups=self.c_iso, bias=False)
        self.dw_hor = nn.Conv2d(self.c_hor, self.c_hor, (1, k_strip), 1, (0, k_strip // 2), groups=self.c_hor, bias=False)
        self.dw_ver = nn.Conv2d(self.c_ver, self.c_ver, (k_strip, 1), 1, (k_strip // 2, 0), groups=self.c_ver, bias=False)
        self.dw_dil = nn.Conv2d(self.c_dil, self.c_dil, 3, 1, dilation, dilation=dilation, groups=self.c_dil, bias=False)

        # ── Frequency half: 3-band TriFreq ──
        c_f = self.c_freq
        self.c_lo = c_f // 3
        self.c_mf = c_f // 3
        self.c_hf = c_f - self.c_lo - self.c_mf

        self.dw_lo = nn.Conv2d(self.c_lo, self.c_lo, k_lo, 1, k_lo // 2, groups=self.c_lo, bias=False)
        self.lp_fine = nn.AvgPool2d(3, 1, 1, count_include_pad=False)
        self.lp_coarse = nn.AvgPool2d(7, 1, 3, count_include_pad=False)
        self.dw_mf = nn.Conv2d(self.c_mf, self.c_mf, 3, 1, 1, groups=self.c_mf, bias=False)
        self.dw_hf = nn.Conv2d(self.c_hf, self.c_hf, k_hi, 1, k_hi // 2, groups=self.c_hf, bias=False)

        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x_dir, x_freq = x[:, :self.c_dir], x[:, self.c_dir:]

        # Direction processing
        d_iso, d_hor, d_ver, d_dil = torch.split(x_dir, [self.c_iso, self.c_hor, self.c_ver, self.c_dil], dim=1)
        y_dir = torch.cat([self.dw_iso(d_iso), self.dw_hor(d_hor), self.dw_ver(d_ver), self.dw_dil(d_dil)], dim=1)

        # Frequency processing
        f_lo, f_mf, f_hf = x_freq[:, :self.c_lo], x_freq[:, self.c_lo:self.c_lo + self.c_mf], x_freq[:, self.c_lo + self.c_mf:]
        y_lo = self.dw_lo(f_lo)
        y_mf = self.dw_mf(self.lp_fine(f_mf) - self.lp_coarse(f_mf))
        y_hf = self.dw_hf(f_hf - self.lp_fine(f_hf))
        y_freq = torch.cat([y_lo, y_mf, y_hf], dim=1)

        out = torch.cat([y_dir, y_freq], dim=1)
        out = _channel_shuffle(out, 2)
        return self.act(self.bn(out))


class NexusPrismGate(nn.Module):
    """Combined NormRatio (sparsity) + FreqContrast (tri-moment) gate."""

    def __init__(self, c, reduction=4):
        super().__init__()
        c_mid = max(c // reduction, 4)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1))  # sparsity weight
        self.lam = nn.Parameter(torch.ones(1, c, 1, 1))   # freq variance weight
        self.fc = nn.Sequential(
            nn.Linear(c, c_mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(c_mid, c, bias=False),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        x_abs = x.abs()
        l1 = x_abs.mean(dim=[2, 3])
        linf = F.adaptive_max_pool2d(x_abs, 1).view(B, C)
        sparsity = l1 / (linf + 1e-6)
        x_sq_mean = x.pow(2).mean(dim=[2, 3])
        x_mean = x.mean(dim=[2, 3])
        freq_var = (x_sq_mean - x_mean.pow(2)).clamp(min=0)
        desc = l1 + self.beta.view(1, C) * sparsity + self.lam.view(1, C) * freq_var
        gate = torch.sigmoid(self.fc(desc))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class NexusPrismBottleneck(nn.Module):
    """1×1 Expand → NexusPrismConv → NexusPrismGate → 1×1 Project → Residual."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=1.0):
        super().__init__()
        c_h = int(c2 * e)
        c_h = max(c_h // 4 * 4, 8)  # ensure divisible by 4 for direction split
        k_strip = k + 2
        dilation = (k - 1) // 2 + 1
        k_lo = k + 2 if k >= 5 else 5
        k_hi = k if k >= 5 else 3
        self.expand = Conv(c1, c_h, 1)
        self.conv = NexusPrismConv(c_h, k_strip=k_strip, dilation=dilation, k_lo=k_lo, k_hi=k_hi)
        self.gate = NexusPrismGate(c_h)
        self.project = Conv(c_h, c2, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.project(self.gate(self.conv(self.expand(x))))
        return x + y if self.add else y


class C3k2_NexusPrism(nn.Module):
    """C2f with NexusPrismBottleneck for backbone."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(NexusPrismBottleneck(self.c, self.c, shortcut, g, k=k) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class NexusPrismCSP(nn.Module):
    """C2f + NexusPrismBottleneck + PolarizedRefine for neck."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(NexusPrismBottleneck(self.c, self.c, shortcut, g, k=k) for _ in range(n))
        # Polarized refinement from Nexus for neck
        self.pos_dw = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.pos_bn = nn.BatchNorm2d(c2)
        self.neg_dw = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.neg_bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        # Polarized contrast refinement
        pos = self.pos_bn(self.pos_dw(F.relu(out)))
        neg = self.neg_bn(self.neg_dw(F.relu(-out)))
        contrast = torch.sigmoid(pos - neg)
        return out * contrast + out


# ═════════════════════════════════════════════════════════════════════════════
# 2. YOLO-PRISM-EDGE — Frequency Intelligence + Lightweight
# ═════════════════════════════════════════════════════════════════════════════

class PrismEdgeConv(nn.Module):
    """
    Partial Frequency Conv: applies TriFreqConv on 1/4 channels, identity on 3/4.
    Combines Edge's PConv sparsity with Prism's frequency decomposition.
    """

    def __init__(self, c, k_lo=5, k_hi=3, n_div=4):
        super().__init__()
        n_div = min(n_div, c)
        self.dim_active = c // n_div
        self.dim_passive = c - self.dim_active

        # TriFreq on active channels
        ca = self.dim_active
        self.c_lo = ca // 3
        self.c_mf = ca // 3
        self.c_hf = ca - self.c_lo - self.c_mf

        if self.c_lo > 0:
            self.dw_lo = nn.Conv2d(self.c_lo, self.c_lo, k_lo, 1, k_lo // 2, groups=self.c_lo, bias=False)
        if self.c_mf > 0:
            self.lp_fine = nn.AvgPool2d(3, 1, 1, count_include_pad=False)
            self.lp_coarse = nn.AvgPool2d(7, 1, 3, count_include_pad=False)
            self.dw_mf = nn.Conv2d(self.c_mf, self.c_mf, 3, 1, 1, groups=self.c_mf, bias=False)
        if self.c_hf > 0:
            self.dw_hf = nn.Conv2d(self.c_hf, self.c_hf, k_hi, 1, k_hi // 2, groups=self.c_hf, bias=False)

        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x_active, x_passive = x[:, :self.dim_active], x[:, self.dim_active:]
        parts = []
        if self.c_lo > 0:
            parts.append(self.dw_lo(x_active[:, :self.c_lo]))
        if self.c_mf > 0:
            mf_in = x_active[:, self.c_lo:self.c_lo + self.c_mf]
            parts.append(self.dw_mf(self.lp_fine(mf_in) - self.lp_coarse(mf_in)))
        if self.c_hf > 0:
            hf_in = x_active[:, self.c_lo + self.c_mf:]
            parts.append(self.dw_hf(hf_in - self.lp_fine(hf_in) if self.c_mf > 0 else hf_in))
        y_active = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.act(self.bn(torch.cat([y_active, x_passive], dim=1)))


class PrismEdgeGate(nn.Module):
    """Lightweight FreqContrastGate for PrismEdge."""

    def __init__(self, c, reduction=4):
        super().__init__()
        c_mid = max(c // reduction, 4)
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(c, c_mid, bias=False), nn.SiLU(inplace=True),
            nn.Linear(c_mid, c, bias=False),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        l1 = x.abs().mean(dim=[2, 3])
        x_sq_mean = x.pow(2).mean(dim=[2, 3])
        l2 = (x_sq_mean + 1e-6).sqrt()
        gamma = l2 / (l1 + 1e-6)
        desc = l1 + self.beta.view(1, C) * gamma
        gate = torch.sigmoid(self.fc(desc))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class PrismEdgeBottleneck(nn.Module):
    """PConv-style partial + TriFreq on active channels + FreqGate."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5):
        super().__init__()
        k_lo = 7 if k >= 5 else 5
        k_hi = 5 if k >= 5 else 3
        self.pconv = PrismEdgeConv(c1, k_lo=k_lo, k_hi=k_hi)
        self.gate = PrismEdgeGate(c1)
        self.cv = Conv(c1, c2, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv(self.gate(self.pconv(x)))
        return x + y if self.add else y


class C3k2_PrismEdge(nn.Module):
    """C2f with PrismEdgeBottleneck for backbone."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(PrismEdgeBottleneck(self.c, self.c, shortcut, g, k=k) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class PrismEdgeCSP(nn.Module):
    """C2f + PrismEdgeBottleneck + GSConv-style neck."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(PrismEdgeBottleneck(self.c, self.c, shortcut, g, k=k) for _ in range(n))
        # Lightweight spatial refinement
        self.refine_dw = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.refine_bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return out * torch.sigmoid(self.refine_bn(self.refine_dw(out))) + out


# ═════════════════════════════════════════════════════════════════════════════
# 3. YOLO-PHOENIX-NEXUS — Multi-Scale + Multi-Direction
# ═════════════════════════════════════════════════════════════════════════════

class PhoenixNexusConv(nn.Module):
    """
    4-path conv combining Phoenix (scale) + Nexus (direction):
    G1: DWConv 3×3 (fine isotropic - Phoenix)
    G2: DWConv 5×5 (coarse isotropic - Phoenix)
    G3: DWConv 1×K (horizontal - Nexus)
    G4: DWConv K×1 (vertical - Nexus)
    """

    def __init__(self, c, k_strip=5):
        super().__init__()
        self.c1 = c // 4
        self.c2 = c // 4
        self.c3 = c // 4
        self.c4 = c - self.c1 - self.c2 - self.c3

        self.dw_fine = nn.Conv2d(self.c1, self.c1, 3, 1, 1, groups=self.c1, bias=False)
        self.dw_coarse = nn.Conv2d(self.c2, self.c2, 5, 1, 2, groups=self.c2, bias=False)
        self.dw_hor = nn.Conv2d(self.c3, self.c3, (1, k_strip), 1, (0, k_strip // 2), groups=self.c3, bias=False)
        self.dw_ver = nn.Conv2d(self.c4, self.c4, (k_strip, 1), 1, (k_strip // 2, 0), groups=self.c4, bias=False)

        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        g1, g2, g3, g4 = torch.split(x, [self.c1, self.c2, self.c3, self.c4], dim=1)
        out = torch.cat([self.dw_fine(g1), self.dw_coarse(g2), self.dw_hor(g3), self.dw_ver(g4)], dim=1)
        out = _channel_shuffle(out, 4)
        return self.act(self.bn(out))


class DualPoolSparsityGate(nn.Module):
    """Combined DualPool (Phoenix) + Sparsity (Nexus) gate."""

    def __init__(self, c, reduction=4):
        super().__init__()
        c_mid = max(c // reduction, 4)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.fc = nn.Sequential(
            nn.Conv2d(c, c_mid, 1, bias=True), nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, x):
        x_abs = x.abs()
        avg = F.adaptive_avg_pool2d(x_abs, 1)
        mx = F.adaptive_max_pool2d(x_abs, 1)
        sparsity = avg / (mx + 1e-6)
        desc = avg + mx + self.beta * sparsity
        return x * self.fc(desc)


class PhoenixNexusBottleneck(nn.Module):
    """1×1 → PhoenixNexusConv → DualPoolSparsityGate → 1×1 → Residual."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=1.0):
        super().__init__()
        c_h = int(c2 * e)
        c_h = max(c_h // 4 * 4, 8)
        k_strip = k + 2
        self.expand = Conv(c1, c_h, 1)
        self.conv = PhoenixNexusConv(c_h, k_strip=k_strip)
        self.gate = DualPoolSparsityGate(c_h)
        self.project = Conv(c_h, c2, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.project(self.gate(self.conv(self.expand(x))))
        return x + y if self.add else y


class C3k2_PhoenixNexus(nn.Module):
    """C2f with PhoenixNexusBottleneck for backbone."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(PhoenixNexusBottleneck(self.c, self.c, shortcut, g, k=k) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class PhoenixNexusCSP(nn.Module):
    """C2f + PhoenixNexusBottleneck + SpatialRefinement for neck."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(PhoenixNexusBottleneck(self.c, self.c, shortcut, g, k=k) for _ in range(n))
        # SRM from Phoenix
        self.srm_dw = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.srm_bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return out * torch.sigmoid(self.srm_bn(self.srm_dw(out))) + out


# ═════════════════════════════════════════════════════════════════════════════
# 4. YOLO-CHIMERA-PRISM — Dilation-Scale + Frequency
# ═════════════════════════════════════════════════════════════════════════════

class ChimeraPrismConv(nn.Module):
    """
    Dual-axis: Trident dilation (Chimera) + DualFreq (Prism).
    Half channels: 3-dilation DWConv, other half: LF+HF frequency split.
    """

    def __init__(self, c, k_lo=5):
        super().__init__()
        self.c_tri = c // 2
        self.c_freq = c - self.c_tri

        # Trident half: 3 dilations
        ct = self.c_tri
        self.ct1 = ct // 3
        self.ct2 = ct // 3
        self.ct3 = ct - self.ct1 - self.ct2
        self.dw_d1 = nn.Conv2d(self.ct1, self.ct1, 3, 1, 1, dilation=1, groups=self.ct1, bias=False)
        self.dw_d2 = nn.Conv2d(self.ct2, self.ct2, 3, 1, 2, dilation=2, groups=self.ct2, bias=False)
        self.dw_d4 = nn.Conv2d(self.ct3, self.ct3, 3, 1, 4, dilation=4, groups=self.ct3, bias=False)

        # Frequency half: LF + HF
        cf = self.c_freq
        self.c_lo = cf // 2
        self.c_hf = cf - self.c_lo
        self.dw_lo = nn.Conv2d(self.c_lo, self.c_lo, k_lo, 1, k_lo // 2, groups=self.c_lo, bias=False)
        self.lp = nn.AvgPool2d(3, 1, 1, count_include_pad=False)
        self.dw_hf = nn.Conv2d(self.c_hf, self.c_hf, 3, 1, 1, groups=self.c_hf, bias=False)

        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x_tri, x_freq = x[:, :self.c_tri], x[:, self.c_tri:]

        # Trident
        t1, t2, t3 = torch.split(x_tri, [self.ct1, self.ct2, self.ct3], dim=1)
        y_tri = torch.cat([self.dw_d1(t1), self.dw_d2(t2), self.dw_d4(t3)], dim=1)

        # Frequency
        f_lo, f_hf = x_freq[:, :self.c_lo], x_freq[:, self.c_lo:]
        y_lo = self.dw_lo(f_lo)
        y_hf = self.dw_hf(f_hf - self.lp(f_hf))
        y_freq = torch.cat([y_lo, y_hf], dim=1)

        out = torch.cat([y_tri, y_freq], dim=1)
        out = _channel_shuffle(out, 2)
        return self.act(self.bn(out))


class SpectralFreqGate(nn.Module):
    """Combined SpectralGate (mean+std) + MCG (L1+concentration)."""

    def __init__(self, c, reduction=4):
        super().__init__()
        c_mid = max(c // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(c, c_mid, bias=False), nn.SiLU(inplace=True),
            nn.Linear(c_mid, c, bias=False),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        mean = x.mean(dim=[2, 3])
        std = x.pow(2).mean(dim=[2, 3]).sub(mean.pow(2)).clamp(min=0).sqrt()
        desc = mean + std
        gate = torch.sigmoid(self.fc(desc))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class ChimeraPrismBottleneck(nn.Module):
    """1×1 → ChimeraPrismConv → SpectralFreqGate → 1×1 → Residual."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=1.0):
        super().__init__()
        c_h = int(c2 * e)
        c_h = max(c_h // 6 * 6, 6)  # divisible by 6 for 3+2 splits
        k_lo = k + 2 if k >= 5 else 5
        self.expand = Conv(c1, c_h, 1)
        self.conv = ChimeraPrismConv(c_h, k_lo=k_lo)
        self.gate = SpectralFreqGate(c_h)
        self.project = Conv(c_h, c2, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.project(self.gate(self.conv(self.expand(x))))
        return x + y if self.add else y


class C3k2_ChimeraPrism(nn.Module):
    """C2f with ChimeraPrismBottleneck for backbone."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(ChimeraPrismBottleneck(self.c, self.c, shortcut, g, k=k) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class ChimeraPrismCSP(nn.Module):
    """C2f + ChimeraPrismBottleneck + CrossScaleModulator for neck."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(ChimeraPrismBottleneck(self.c, self.c, shortcut, g, k=k) for _ in range(n))
        # Cross-Scale Modulator from Chimera
        self.csm_content = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True),
        )
        self.csm_detail = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.csm_bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        _, _, H, W = out.shape
        content = F.adaptive_avg_pool2d(out, (max(H // 2, 1), max(W // 2, 1)))
        content = self.csm_content(content)
        content = F.interpolate(content, size=(H, W), mode='bilinear', align_corners=False)
        detail = self.csm_bn(self.csm_detail(out))
        return torch.sigmoid(content) * detail + detail


# ═════════════════════════════════════════════════════════════════════════════
# 5. YOLO-SPECTRA-EDGE — Wavelet Full-Spectrum + Lightweight
# ═════════════════════════════════════════════════════════════════════════════

class SpectraEdgeConv(nn.Module):
    """
    Partial Wavelet Conv: Haar wavelet on 1/4 channels (active), identity on 3/4.
    Combines Spectra's wavelet decomposition with Edge's PConv efficiency.
    """

    def __init__(self, c, k=3, n_div=4):
        super().__init__()
        n_div = min(n_div, c)
        self.dim_active = c // n_div
        self.dim_passive = c - self.dim_active

        ca = self.dim_active
        # Wavelet subband processing at half resolution
        self.c_ll = ca // 4
        self.c_lh = ca // 4
        self.c_hl = ca // 4
        self.c_hh = ca - self.c_ll - self.c_lh - self.c_hl

        if self.c_ll > 0:
            self.dw_ll = nn.Conv2d(self.c_ll, self.c_ll, k, 1, k // 2, groups=self.c_ll, bias=False)
        if self.c_lh > 0:
            self.dw_lh = nn.Conv2d(self.c_lh, self.c_lh, (1, k), 1, (0, k // 2), groups=self.c_lh, bias=False)
        if self.c_hl > 0:
            self.dw_hl = nn.Conv2d(self.c_hl, self.c_hl, (k, 1), 1, (k // 2, 0), groups=self.c_hl, bias=False)
        if self.c_hh > 0:
            self.dw_hh = nn.Conv2d(self.c_hh, self.c_hh, 3, 1, 1, groups=self.c_hh, bias=False)

        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    @staticmethod
    def _haar_forward(x):
        """2D Haar wavelet forward transform."""
        # Pad if odd
        _, _, H, W = x.shape
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))
        x_even_h = x[:, :, 0::2, :]
        x_odd_h = x[:, :, 1::2, :]
        l = x_even_h + x_odd_h
        h = x_even_h - x_odd_h
        ll = l[:, :, :, 0::2] + l[:, :, :, 1::2]
        lh = l[:, :, :, 0::2] - l[:, :, :, 1::2]
        hl = h[:, :, :, 0::2] + h[:, :, :, 1::2]
        hh = h[:, :, :, 0::2] - h[:, :, :, 1::2]
        return ll * 0.5, lh * 0.5, hl * 0.5, hh * 0.5

    @staticmethod
    def _haar_inverse(ll, lh, hl, hh, target_h, target_w):
        """2D Haar wavelet inverse transform."""
        l_even = ll + lh
        l_odd = ll - lh
        h_even = hl + hh
        h_odd = hl - hh
        _, C, Hh, Wh = ll.shape
        l = torch.zeros(ll.shape[0], C, Hh, Wh * 2, device=ll.device, dtype=ll.dtype)
        l[:, :, :, 0::2] = l_even
        l[:, :, :, 1::2] = l_odd
        h = torch.zeros_like(l)
        h[:, :, :, 0::2] = h_even
        h[:, :, :, 1::2] = h_odd
        out = torch.zeros(ll.shape[0], C, Hh * 2, Wh * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, :] = l + h
        out[:, :, 1::2, :] = l - h
        return out[:, :, :target_h, :target_w] * 0.5

    def forward(self, x):
        _, _, H, W = x.shape
        x_active, x_passive = x[:, :self.dim_active], x[:, self.dim_active:]

        # Split active channels into 4 subband groups
        a_ll = x_active[:, :self.c_ll]
        a_lh = x_active[:, self.c_ll:self.c_ll + self.c_lh]
        a_hl = x_active[:, self.c_ll + self.c_lh:self.c_ll + self.c_lh + self.c_hl]
        a_hh = x_active[:, self.c_ll + self.c_lh + self.c_hl:]

        parts = []
        if self.c_ll > 0:
            ll, _, _, _ = self._haar_forward(a_ll)
            ll = self.dw_ll(ll)
            parts.append(self._haar_inverse(ll, torch.zeros_like(ll), torch.zeros_like(ll), torch.zeros_like(ll), H, W))
        if self.c_lh > 0:
            _, lh, _, _ = self._haar_forward(a_lh)
            lh = self.dw_lh(lh)
            parts.append(self._haar_inverse(torch.zeros_like(lh), lh, torch.zeros_like(lh), torch.zeros_like(lh), H, W))
        if self.c_hl > 0:
            _, _, hl, _ = self._haar_forward(a_hl)
            hl = self.dw_hl(hl)
            parts.append(self._haar_inverse(torch.zeros_like(hl), torch.zeros_like(hl), hl, torch.zeros_like(hl), H, W))
        if self.c_hh > 0:
            _, _, _, hh = self._haar_forward(a_hh)
            hh = self.dw_hh(hh)
            parts.append(self._haar_inverse(torch.zeros_like(hh), torch.zeros_like(hh), torch.zeros_like(hh), hh, H, W))

        y_active = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.act(self.bn(torch.cat([y_active, x_passive], dim=1)))


class SpectraEdgeBottleneck(nn.Module):
    """PConv-style partial wavelet + 1×1 mixing."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5):
        super().__init__()
        self.seconv = SpectraEdgeConv(c1, k=k)
        self.cv = Conv(c1, c2, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv(self.seconv(x))
        return x + y if self.add else y


class C3k2_SpectraEdge(nn.Module):
    """C2f with SpectraEdgeBottleneck for backbone."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(SpectraEdgeBottleneck(self.c, self.c, shortcut, g, k=k) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SpectraEdgeCSP(nn.Module):
    """C2f + SpectraEdgeBottleneck + lightweight GSConv neck."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(SpectraEdgeBottleneck(self.c, self.c, shortcut, g, k=k) for _ in range(n))
        self.refine_dw = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.refine_bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return out * torch.sigmoid(self.refine_bn(self.refine_dw(out))) + out
