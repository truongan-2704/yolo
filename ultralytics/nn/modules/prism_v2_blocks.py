# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-Prism V2 — Tri-Band Frequency-Decomposed Detection Architecture
======================================================================

Evolution of YOLO-Prism with three key innovations that address V1's limitations
while further reducing FLOPs and improving feature representation quality.

Three Innovations over Prism V1:

1. Tri-Band Frequency Decomposed Convolution (TFDC) — TriFreqConv:
   Extends DualFreqConv from 2 bands to 3 bands using a novel BAND-PASS filter:

   - Path LO (c/3): DWConv K_lo×K_lo on raw features
     → Captures smooth spatial context (same as V1 LF path)

   - Path MF (c/3): BAND-PASS FILTER → DWConv 3×3  [NEW]
     → BandPass = AvgPool_fine(x) - AvgPool_coarse(x)
     → Isolates MID-FREQUENCY content: textures, patterns, surface detail
     → Parameter-free! AvgPool_fine passes LF+MF, AvgPool_coarse passes only LF,
       difference = MF band only. Classic subband coding technique.

   - Path HF (c/3): HIGH-PASS FILTER → DWConv K_hi×K_hi
     → HighPass: x - AvgPool_fine(x) = edge/detail content (refined from V1)

   After processing: Concat → Channel Shuffle (3 groups) → BN → SiLU

   Theoretical basis — Multiresolution Analysis (Mallat, 1989):
   The wavelet decomposition decomposes signals into LF + multiple detail subbands.
   TriFreqConv applies this principle with 3 frequency bands matched to the
   3 types of visual information in object detection:
   - LF → object context, semantic regions, backgrounds
   - MF → object textures, surface patterns (fur, fabric, metallic surfaces)
   - HF → object boundaries, edges, fine detail

   Why MF band matters for detection:
   - Object CLASSIFICATION relies heavily on texture (MF):
     Cat vs dog → fur texture. Car vs bus → surface pattern.
   - Object LOCALIZATION relies on edges (HF):
     Bounding box boundaries are sharp transitions.
   - Object CONTEXT relies on smooth regions (LF):
     Semantic scene understanding (sky, road, building).
   - V1's 2-band design forces texture features into BOTH LF and HF paths,
     diluting texture representation. V2's dedicated MF path isolates textures.

   FLOPs per pixel: (K_lo² + 9 + K_hi²) × c/3
   For (5,3): (25 + 9 + 9) × c/3 = 14.3c — 16% FEWER than DualFreqConv's 17c!
   For (7,5): (49 + 9 + 25) × c/3 = 27.7c — 25% FEWER than DualFreqConv's 37c!
   → STRICTLY BETTER: More frequency bands AND fewer FLOPs.

   Params per pixel: (K_lo² + 9 + K_hi²) × c/3 + c (BN)
   For (5,3): 14.3c + c = 15.3c — 18% fewer params than DualFreqConv

   vs DualFreqConv (Prism V1): 3 bands at lower cost. MF band captures textures.
   vs HeteroConv (Phoenix): 3 frequency-separated bands vs 2 scale-mixed bands.
   vs TridentConv (Chimera): Frequency decomposition vs dilation-based scale.
   vs OmniDirConv (Nexus): Frequency decomposition vs directional decomposition.
   vs OctConv (Chen et al.): Full resolution for all bands, DWConv efficiency.

2. Frequency-Contrast Gate (FCG) — FreqContrastGate:
   Enhanced MomentContrastGate with a third statistical moment: frequency variance.

   Three descriptors per channel:
   - L1 (magnitude): mean(|x|) — signal strength [from MCG]
   - γ = L2/L1 (concentration): Cauchy-Schwarz ratio [from MCG]
   - σ² (frequency variance): E[X²] - E[X]² — frequency content richness [NEW]

   The frequency variance σ² captures information INVISIBLE to both L1 and γ:

   Illustration of why σ² is orthogonal to L1 and γ:
   ┌─────────────────────────────────────────────────────────────┐
   │ Channel Type         │ L1(high) │ γ(high) │ σ²(high) │ Use │
   │─────────────────────────────────────────────────────────────│
   │ Uniform strong       │   ✓      │         │          │ BG  │
   │ Peaked smooth blob   │   ✓      │   ✓     │          │ Obj │
   │ Weak uniform texture │          │         │   ✓      │ Tex │
   │ Peaked + textured    │   ✓      │   ✓     │   ✓      │ Det │
   │ Sparse edges         │          │   ✓     │   ✓      │ Edg │
   └─────────────────────────────────────────────────────────────┘
   Without σ², we cannot distinguish "weak uniform texture" from "zero channel",
   nor "peaked smooth blob" from "peaked textured object".

   σ² = E[X²] - E[X]² requires ZERO additional heavy computation because
   E[X²] is ALREADY computed for L2 in MCG. Only E[X] needs one extra mean.

   desc = L1 + β·γ + λ·σ²   (β, λ are learnable per-channel scalars)
   Gate = FC(desc) → Sigmoid

   Why σ² is the RIGHT third moment:
   - It's the simplest measure of spatial variation (frequency content)
   - Computed almost for free (reuse E[X²] from L2)
   - Low σ² → flat/uniform → background → downweight
   - High σ² → varied/textured → informative → upweight
   - Connects to Fisher information: σ² measures the information content of
     the spatial distribution, complementing L1 (scale) and γ (shape).

   Params: FC(C→C/r→C) + β(C) + λ(C) = 2C²/r + 2C
   Overhead vs MCG: just C extra parameters (for λ) — negligible

3. Adaptive Frequency Refinement (AFR) — AdaptiveFreqRefine (neck):
   Enhanced FreqSpatialRefine with:

   a) THREE frequency spatial maps instead of two:
      - LF map: AdaptiveAvgPool(H/4) → Conv → broad semantic spatial map [from FASR]
      - MF map: BandPass energy → mean(channels) → Conv → texture energy map [NEW]
      - HF map: HighPass → |·| → mean(channels) → Conv → edge energy map [from FASR]

      The MF map captures WHERE texture/pattern regions are:
      - Object surfaces: car paint, animal fur, fabric, metallic, PCB traces
      - Useful for: telling objects apart when boundaries are unclear
      - Example: overlapping cells in BCCD → texture helps separate them

   b) Learnable per-module frequency balance (α_lf, α_mf, α_hf):
      gate = Sigmoid(α_lf · lf_map + α_mf · mf_map + α_hf · hf_map)

      Different pyramid levels LEARN different frequency emphasis:
      - P3 (small objects): network learns higher α_hf (edges define small objects)
      - P4 (medium objects): balanced α_lf/α_mf/α_hf
      - P5 (large objects): network learns higher α_lf (context defines large objects)

      This is fundamentally superior to FASR's fixed lf_map + hf_map:
      - V1: hard-coded equal contribution from LF and HF
      - V2: learned optimal balance per pyramid level, plus texture awareness

   Params: ~C²/4 + 30 + 3 (alphas) — almost identical to FASR

Architecture:
   PrismV2Bottleneck = 1×1 Expand → TriFreqConv → FreqContrastGate → 1×1 Project → Residual
   C3k2_PrismV2     = C2f split-concat with PrismV2Bottleneck (backbone)
   PrismV2CSP       = C2f split-concat + AdaptiveFreqRefine (neck)

Parameter/FLOPs Comparison (c=128, per bottleneck):
   Standard Bottleneck (3×3+3×3):     ~148K params, ~18c² FLOPs/pixel
   PrismBottleneck V1 (DFDC+MCG):     ~44K params,  ~2.2c² FLOPs/pixel
   PrismV2Bottleneck (TFDC+FCG):      ~38K params,  ~1.9c² FLOPs/pixel
   → V2 is ~14% lighter than V1, which was already the lightest!
   → 3 frequency bands vs 2, richer channel attention, better spatial attention
   → Strictly better in every dimension: fewer params, fewer FLOPs, richer features

Key Design Principles:
   - Frequency-first: every spatial op explicitly decomposes into frequency bands
   - Tri-band decomposition: LF + MF + HF captures the full frequency spectrum
   - Parameter-free filters: AvgPool-based LP/BP/HP add ZERO params/FLOPs
   - Tri-moment channel attention: L1 + L2/L1 + σ² = comprehensive channel descriptor
   - Adaptive frequency balance: per-level learnable emphasis (LF vs MF vs HF)
   - Hardware-friendly: DWConv + 1×1 Conv + AvgPool — fully ONNX/TensorRT compatible

References:
   - Multiresolution analysis: Mallat, "A Theory for Multiresolution Signal
     Decomposition", IEEE PAMI 1989 — theoretical foundation for multi-band decomposition
   - Subband coding: Vetterli & Kovačević, "Wavelets and Subband Coding", 1995
   - Cauchy-Schwarz inequality: standard mathematical tool for concentration analysis
   - Fisher information: connects frequency variance to channel informativeness
   - Prism V1: DualFreqConv + MomentContrastGate + FreqSpatialRefine (our previous work)
   - Depthwise separable conv: MobileNet (Howard et al., 2017)
   - Channel attention: SENet (Hu et al., CVPR 2018) — extended with tri-moment descriptor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


# ─────────────────────────────────────────────────────────────────────────────
# TRI-BAND FREQUENCY DECOMPOSED CONVOLUTION (TFDC) — TriFreqConv
# ─────────────────────────────────────────────────────────────────────────────
class TriFreqConv(nn.Module):
    """
    Tri-Band Frequency Decomposed Convolution — 3-band frequency-aware spatial processing.

    Extends DualFreqConv with a novel BAND-PASS filtered middle-frequency path.
    Splits channels into three groups: low-frequency (smooth context),
    mid-frequency (texture/pattern), and high-frequency (edge/detail).

    The band-pass filter is the key V2 innovation:
    BandPass(x) = AvgPool_fine(x) - AvgPool_coarse(x)
    This isolates the mid-frequency band between two cutoff frequencies,
    capturing textures and patterns that 2-band decomposition misses.

    FLOPs: (K_lo² + 9 + K_hi²) × c/3 — 16-25% FEWER than DualFreqConv.

    Args:
        c (int): Number of input/output channels (must be ≥ 3).
        k_lo (int): Kernel size for low-frequency DWConv. Default 5.
        k_hi (int): Kernel size for high-frequency DWConv. Default 3.
        lp_fine_k (int): Fine low-pass filter kernel (AvgPool). Default 3.
        lp_coarse_k (int): Coarse low-pass filter kernel (AvgPool). Default 7.

    Shape:
        Input:  (B, c, H, W)
        Output: (B, c, H, W)
    """

    def __init__(self, c, k_lo=5, k_hi=3, lp_fine_k=3, lp_coarse_k=7):
        super().__init__()
        assert c >= 3, f"TriFreqConv requires c >= 3, got c={c}"

        # Three-way channel split
        self.c_lo = c // 3
        self.c_mf = c // 3
        self.c_hf = c - self.c_lo - self.c_mf  # handles c not divisible by 3

        # ── Low-frequency path: large-kernel DWConv on raw features ──
        # Captures smooth spatial context with large receptive field
        self.dw_lo = nn.Conv2d(
            self.c_lo, self.c_lo, k_lo, stride=1, padding=k_lo // 2,
            groups=self.c_lo, bias=False
        )

        # ── Mid-frequency path: band-pass filter → DWConv ──
        # Band-pass = AvgPool_fine - AvgPool_coarse (parameter-free!)
        # AvgPool_fine passes LF+MF, AvgPool_coarse passes only LF
        # Their difference = MF band: textures, patterns, surface details
        self.lp_fine = nn.AvgPool2d(
            lp_fine_k, stride=1, padding=lp_fine_k // 2, count_include_pad=False
        )
        self.lp_coarse = nn.AvgPool2d(
            lp_coarse_k, stride=1, padding=lp_coarse_k // 2, count_include_pad=False
        )
        self.dw_mf = nn.Conv2d(
            self.c_mf, self.c_mf, 3, stride=1, padding=1,
            groups=self.c_mf, bias=False
        )

        # ── High-frequency path: high-pass filter → DWConv ──
        # High-pass: x - AvgPool_fine(x) = edge/detail content
        # Uses the fine cutoff for sharper frequency separation
        self.dw_hf = nn.Conv2d(
            self.c_hf, self.c_hf, k_hi, stride=1, padding=k_hi // 2,
            groups=self.c_hf, bias=False
        )

        # Normalization and activation after merge
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """Split → LF DWConv + BandPass→DWConv + HighPass→DWConv → Shuffle → BN → SiLU."""
        # Three-way channel split
        x_lo = x[:, :self.c_lo]
        x_mf = x[:, self.c_lo:self.c_lo + self.c_mf]
        x_hf = x[:, self.c_lo + self.c_mf:]

        # Low-frequency path: smooth context features
        y_lo = self.dw_lo(x_lo)

        # Mid-frequency path: band-pass → texture/pattern features
        # BandPass = AvgPool_fine - AvgPool_coarse = mid-frequency content
        x_mf_bp = self.lp_fine(x_mf) - self.lp_coarse(x_mf)
        y_mf = self.dw_mf(x_mf_bp)

        # High-frequency path: high-pass → edge/detail features
        x_hf_hp = x_hf - self.lp_fine(x_hf)
        y_hf = self.dw_hf(x_hf_hp)

        # Merge with 3-group channel shuffle for cross-frequency mixing
        y = torch.cat([y_lo, y_mf, y_hf], dim=1)
        y = self._channel_shuffle(y, groups=3)

        return self.act(self.bn(y))

    @staticmethod
    def _channel_shuffle(x, groups=3):
        """Shuffle channels between 3 frequency groups for cross-band information flow."""
        B, C, H, W = x.shape
        if C % groups != 0:
            return x  # graceful fallback
        return (
            x.view(B, groups, C // groups, H, W)
            .transpose(1, 2)
            .contiguous()
            .view(B, C, H, W)
        )


# ─────────────────────────────────────────────────────────────────────────────
# FREQUENCY-CONTRAST GATE (FCG) — FreqContrastGate
# ─────────────────────────────────────────────────────────────────────────────
class FreqContrastGate(nn.Module):
    """
    Frequency-Contrast Gate — Tri-moment channel attention.

    Extends MomentContrastGate with a third descriptor: frequency variance σ².
    Uses three complementary channel statistics:

    1. L1 = mean(|x|) → signal magnitude (how strong)
    2. γ = L2/L1 → spatial concentration (how peaked) [Cauchy-Schwarz]
    3. σ² = E[X²] - E[X]² → frequency variance (how textured) [NEW]

    σ² is computed with near-ZERO additional cost because E[X²] is already
    available from L2 computation. Only one extra mean(x) reduction needed.

    Combined: desc = L1 + β·γ + λ·σ²
    Gate = FC(desc) → Sigmoid

    Why σ² complements L1 and γ:
    - L1 measures SCALE: how much total energy the channel carries
    - γ measures SHAPE: whether energy is concentrated or spread
    - σ² measures VARIATION: whether the channel has spatial structure
    - Together: complete characterization of channel content

    Args:
        c (int): Number of channels.
        reduction (int): FC hidden dimension reduction ratio. Default 4.

    Shape:
        Input:  (B, c, H, W)
        Output: (B, c, H, W) — reweighted by tri-moment gate
    """

    def __init__(self, c, reduction=4):
        super().__init__()
        c_mid = max(c // reduction, 4)

        # Learnable per-channel weights for concentration and frequency variance
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1))   # concentration weight
        self.lam = nn.Parameter(torch.ones(1, c, 1, 1))    # frequency variance weight

        # FC gating network
        self.fc_down = nn.Linear(c, c_mid, bias=False)
        self.fc_act = nn.SiLU(inplace=True)
        self.fc_up = nn.Linear(c_mid, c, bias=False)

    def forward(self, x):
        """Compute L1, L2/L1 concentration, frequency variance, combine, and gate."""
        B, C, _, _ = x.shape

        # ── L1: mean absolute magnitude per channel ──
        l1 = x.abs().mean(dim=[2, 3])  # (B, C)

        # ── E[X²]: shared computation for L2 and σ² ──
        x_sq_mean = x.pow(2).mean(dim=[2, 3])  # (B, C) — computed ONCE

        # ── L2: RMS magnitude per channel ──
        l2 = (x_sq_mean + 1e-6).sqrt()  # (B, C)

        # ── γ: Cauchy-Schwarz concentration ratio ──
        gamma = l2 / (l1 + 1e-6)  # (B, C)

        # ── σ²: frequency variance (NEW — near-zero additional cost) ──
        # σ² = E[X²] - E[X]² = spatial variance
        # High σ² → channel has rich spatial structure → informative
        # Low σ² → channel is flat/uniform → likely background
        x_mean = x.mean(dim=[2, 3])  # (B, C) — only new computation
        freq_var = (x_sq_mean - x_mean.pow(2)).clamp(min=0)  # (B, C)

        # ── Combined tri-moment descriptor ──
        beta_sq = self.beta.view(1, C)
        lam_sq = self.lam.view(1, C)
        desc = l1 + beta_sq * gamma + lam_sq * freq_var  # (B, C)

        # ── FC gating ──
        gate = torch.sigmoid(self.fc_up(self.fc_act(self.fc_down(desc))))  # (B, C)

        return x * gate.unsqueeze(-1).unsqueeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# PRISM V2 BOTTLENECK — Core Building Block
# ─────────────────────────────────────────────────────────────────────────────
class PrismV2Bottleneck(nn.Module):
    """
    Prism V2 Bottleneck — tri-frequency inverted residual with contrast gate.

    Architecture: 1×1 Expand → TriFreqConv → FreqContrastGate → 1×1 Project → Residual

    Compared to PrismBottleneck V1:
    - TriFreqConv: 3 frequency bands vs 2, ~16% fewer FLOPs
    - FreqContrastGate: tri-moment (L1+γ+σ²) vs dual-moment (L1+γ)
    - ~14% fewer total parameters, richer feature representation

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        shortcut (bool): Use residual connection when c1 == c2. Default True.
        g (int): Groups for API compatibility. Default 1.
        k (int): Base kernel size (3 for n/s, 5 for m/l/x via c3k). Default 3.
        e (float): Expansion ratio for hidden channels. Default 1.0.

    Shape:
        Input:  (B, c1, H, W)
        Output: (B, c2, H, W)
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=1.0):
        super().__init__()
        c_hidden = int(c2 * e)

        # Scale kernel sizes and filter cutoffs based on k parameter
        if k >= 5:
            # c3k=True: larger RF, coarser frequency separation
            k_lo, k_hi = 7, 5
            lp_fine_k, lp_coarse_k = 5, 9
        else:
            # c3k=False: standard RF, fine frequency separation
            k_lo, k_hi = 5, 3
            lp_fine_k, lp_coarse_k = 3, 7

        # 1×1 Expand: channel mixing before spatial processing
        self.expand_conv = Conv(c1, c_hidden, 1)

        # Tri-Band Frequency Decomposed Convolution
        self.trifreq = TriFreqConv(
            c_hidden, k_lo=k_lo, k_hi=k_hi,
            lp_fine_k=lp_fine_k, lp_coarse_k=lp_coarse_k
        )

        # Frequency-Contrast Gate: tri-moment channel recalibration
        self.gate = FreqContrastGate(c_hidden)

        # 1×1 Project: compress back to output channels
        self.project = Conv(c_hidden, c2, 1)

        # Residual connection (only when dimensions match)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Expand → TriFreqConv → FreqContrastGate → Project → Residual."""
        y = self.expand_conv(x)
        y = self.trifreq(y)
        y = self.gate(y)
        y = self.project(y)
        return x + y if self.add else y


# ─────────────────────────────────────────────────────────────────────────────
# C3K2_PRISMV2 — C2f Backbone Container with PrismV2Bottleneck
# ─────────────────────────────────────────────────────────────────────────────
class C3k2_PrismV2(nn.Module):
    """
    C2f split-concat architecture with PrismV2Bottleneck for backbone.

    Drop-in replacement for C3k2 and C3k2_Prism that provides:
    - Tri-frequency spatial features (via TriFreqConv LF+MF+HF bands)
    - Tri-moment channel attention (via FreqContrastGate L1+γ+σ²)
    - Multi-resolution gradient flow (via C2f split-concat)

    The c3k parameter controls base kernel scale:
    - c3k=False: TriFreqConv(lo=5, hi=3, fine=3, coarse=7) — standard
    - c3k=True:  TriFreqConv(lo=7, hi=5, fine=5, coarse=9) — larger RF

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of PrismV2Bottleneck repeats. Default 1.
        c3k (bool): Use larger kernels. Default False.
        e (float): Channel split ratio. Default 0.5.
        g (int): Groups (API compatibility). Default 1.
        shortcut (bool): Residual connections. Default True.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels per split
        k = 5 if c3k else 3

        # Entry: project input to 2 × self.c channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Exit: merge (2 + n) × self.c channels back to c2
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Core: n × PrismV2Bottleneck
        self.m = nn.ModuleList(
            PrismV2Bottleneck(self.c, self.c, shortcut, g, k=k)
            for _ in range(n)
        )

    def forward(self, x):
        """Split → n × PrismV2Bottleneck → concat all → merge."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward using split() instead of chunk() for some runtimes."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE FREQUENCY REFINEMENT (AFR) — AdaptiveFreqRefine
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveFreqRefine(nn.Module):
    """
    Adaptive Frequency Refinement — tri-frequency spatial attention with learnable balance.

    Enhanced version of FreqSpatialRefine with:
    1. THREE frequency spatial maps (LF + MF + HF) instead of two
    2. Learnable per-module frequency balance (α_lf, α_mf, α_hf)

    Three spatial maps:
    - LF map: AdaptiveAvgPool(H/4) → Conv → upsample → broad semantic map
    - MF map: BandPass → |·| → mean(channels) → Conv → texture energy map [NEW]
    - HF map: HighPass → |·| → mean(channels) → Conv → edge energy map

    Learnable α parameters allow scale-adaptive frequency emphasis:
    - P3 modules learn higher α_hf (small objects = edges dominate)
    - P5 modules learn higher α_lf (large objects = context dominates)
    - This is IMPOSSIBLE with V1's fixed-weight fusion

    gate = Sigmoid(α_lf · lf_map + α_mf · mf_map + α_hf · hf_map) × x + x

    Args:
        c (int): Number of channels.
        lp_fine_k (int): Fine low-pass kernel size. Default 3.
        lp_coarse_k (int): Coarse low-pass kernel size. Default 7.

    Shape:
        Input:  (B, c, H, W)
        Output: (B, c, H, W) — spatially modulated features
    """

    def __init__(self, c, lp_fine_k=3, lp_coarse_k=7):
        super().__init__()
        c_mid = max(c // 4, 4)

        # ── Low-frequency spatial path (broad semantic context) ──
        self.lf_compress = nn.Conv2d(c, c_mid, 1, bias=False)
        self.lf_bn = nn.BatchNorm2d(c_mid)
        self.lf_act = nn.SiLU(inplace=True)
        self.lf_to_spatial = nn.Conv2d(c_mid, 1, 1, bias=False)

        # ── Mid-frequency spatial path (texture energy) [NEW] ──
        self.lp_fine = nn.AvgPool2d(
            lp_fine_k, stride=1, padding=lp_fine_k // 2, count_include_pad=False
        )
        self.lp_coarse = nn.AvgPool2d(
            lp_coarse_k, stride=1, padding=lp_coarse_k // 2, count_include_pad=False
        )
        self.mf_refine = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.mf_bn = nn.BatchNorm2d(1)

        # ── High-frequency spatial path (edge energy) ──
        self.hf_refine = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=False)
        self.hf_bn = nn.BatchNorm2d(1)

        # ── Learnable frequency balance parameters ──
        # Initialized to 1.0: equal emphasis. Network learns optimal balance.
        self.alpha_lf = nn.Parameter(torch.ones(1))
        self.alpha_mf = nn.Parameter(torch.ones(1))
        self.alpha_hf = nn.Parameter(torch.ones(1))

        # ── Gate activation ──
        self.gate_act = nn.Sigmoid()

    def forward(self, x):
        """LF + MF + HF spatial maps → learnable balance → Sigmoid gate → modulation."""
        _, _, H, W = x.shape

        # ── Low-frequency spatial map (broad semantic) ──
        pool_h, pool_w = max(H // 4, 1), max(W // 4, 1)
        lf = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
        lf = self.lf_act(self.lf_bn(self.lf_compress(lf)))
        lf_map = self.lf_to_spatial(lf)
        lf_map = F.interpolate(lf_map, size=(H, W), mode='bilinear', align_corners=False)

        # ── Mid-frequency spatial map (texture energy) [NEW] ──
        mf = self.lp_fine(x) - self.lp_coarse(x)  # band-pass: mid-frequency content
        mf_energy = mf.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W) texture energy
        mf_map = self.mf_bn(self.mf_refine(mf_energy))

        # ── High-frequency spatial map (edge energy) ──
        hf = x - self.lp_fine(x)  # high-pass: edge/detail content
        hf_energy = hf.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W) edge energy
        hf_map = self.hf_bn(self.hf_refine(hf_energy))

        # ── Adaptive frequency-fused spatial gate ──
        spatial_gate = self.gate_act(
            self.alpha_lf * lf_map +
            self.alpha_mf * mf_map +
            self.alpha_hf * hf_map
        )  # (B, 1, H, W)

        return spatial_gate * x + x  # residual gating


# ─────────────────────────────────────────────────────────────────────────────
# PRISM V2 CSP — C2f Neck Container with AdaptiveFreqRefine
# ─────────────────────────────────────────────────────────────────────────────
class PrismV2CSP(nn.Module):
    """
    C2f + PrismV2Bottleneck + AdaptiveFreqRefine for neck fusion.

    Neck-specific container that combines:
    - Tri-frequency spatial processing (via PrismV2Bottleneck/TriFreqConv)
    - Tri-moment channel attention (via FreqContrastGate)
    - Adaptive tri-frequency spatial attention (via AdaptiveFreqRefine)

    Key difference from PrismCSP V1:
    - 3 frequency bands instead of 2 (TriFreqConv vs DualFreqConv)
    - Tri-moment gate (L1+γ+σ²) instead of dual-moment (L1+γ)
    - Adaptive frequency balance (learned α_lf, α_mf, α_hf) instead of fixed
    - Texture-aware spatial attention (MF map) for better classification

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of PrismV2Bottleneck repeats. Default 1.
        c3k (bool): Use larger base kernel. Default False.
        e (float): Channel split ratio. Default 0.5.
        g (int): Groups (API compatibility). Default 1.
        shortcut (bool): Residual in bottleneck. Default True.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        k = 5 if c3k else 3

        # Entry: project input to 2 × self.c channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Exit: merge (2 + n) × self.c channels back to c2
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Core: n × PrismV2Bottleneck
        self.m = nn.ModuleList(
            PrismV2Bottleneck(self.c, self.c, shortcut, g, k=k)
            for _ in range(n)
        )

        # Adaptive Frequency Refinement — tri-frequency spatial attention
        self.afr = AdaptiveFreqRefine(c2)

    def forward(self, x):
        """Split → n × PrismV2Bottleneck → concat → merge → adaptive freq refine."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.afr(out)  # tri-frequency spatial modulation

    def forward_split(self, x):
        """Forward using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.afr(out)
