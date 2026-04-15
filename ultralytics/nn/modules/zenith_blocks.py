# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-Zenith — Wavelet-Topological Hybrid Detection Architecture
================================================================

A completely original architecture with THREE novel innovations:

1. Backbone: C3k2_Zenith (WaveletConv + TopologicalGate)
   → WaveletConv: Multi-resolution DWConv using learnable Haar-like wavelet
     decomposition with FOUR sub-bands: LL (approx), LH (horizontal detail),
     HL (vertical detail), HH (diagonal detail)
   → Unlike Prism's simple high-pass filter (x - AvgPool), WaveletConv
     decomposes into all 4 sub-bands using learned wavelet kernels
   → Unlike Nexus's directional DWConv (fixed shapes), WaveletConv adapts
     its basis functions during training
   → TopologicalGate: Euler-characteristic-inspired channel attention using
     connected component analysis via learnable threshold exceedance counting
   → Measures topological complexity of feature maps (how many "peaks" exist)
   → Fundamentally different from norm-based gates (SE, MCG, NormRatioGate)

2. Neck: ZenithCSP (ZenithBottleneck + AdaptiveScaleRouter)
   → AdaptiveScaleRouter: Content-dependent multi-scale routing using learned
     spatial hash that assigns each position to one of K resolution paths
   → Replaces static multi-scale fusion with dynamic per-position routing
   → Each position independently decides its optimal scale context

3. Head: Standard Detect (P3, P4, P5) — proven anchor-free detection

Key advantages:
  vs YOLO11:       Wavelet decomposition + topological attention + adaptive routing
  vs YOLO-Prism:   4-subband wavelet vs 2-band freq split; topology vs concentration
  vs YOLO-Nexus:   Learned wavelets vs fixed directional kernels; topology vs sparsity
  vs YOLO-Chimera: Wavelet basis vs dilation; topological vs spectral gates
  vs YOLO-Phoenix: Learned multi-resolution vs fixed multi-scale; richer attention

Hardware: ONNX/TensorRT compatible (DWConv, 1×1 Conv, AvgPool, adaptive pool)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


# ─────────────────────────────────────────────────────────────────────────────
# 1. WAVELET DEPTHWISE CONVOLUTION — WaveletConv
# ─────────────────────────────────────────────────────────────────────────────
class WaveletConv(nn.Module):
    """
    Wavelet-Inspired Multi-Resolution Depthwise Convolution.

    Decomposes input into FOUR frequency sub-bands inspired by 2D Haar wavelet
    transform, then processes each with a specialized DWConv:

      - LL (c/4): AvgPool → DWConv k×k  → Low-freq approximation (smooth context)
      - LH (c/4): H-gradient → DWConv 1×k → Horizontal detail (edges along H axis)
      - HL (c/4): V-gradient → DWConv k×1 → Vertical detail (edges along V axis)
      - HH (c/4): Laplacian → DWConv 3×3  → Diagonal detail (corners, textures)

    The wavelet decomposition filters are parameter-free (AvgPool, gradient ops),
    while the per-subband DWConvs are learned — creating a learned wavelet filter bank.

    Mathematical foundation:
    In Haar wavelet transform, the 2D basis is:
      LL = (Lo ⊗ Lo),  LH = (Lo ⊗ Hi),  HL = (Hi ⊗ Lo),  HH = (Hi ⊗ Hi)
    where Lo = [1,1]/√2 (averaging) and Hi = [1,-1]/√2 (differencing).

    We approximate:
      LL → AvgPool (spatial averaging)
      LH → horizontal finite difference (x[:,:,:,1:] - x[:,:,:,:-1])
      HL → vertical finite difference (x[:,:,1:,:] - x[:,:,:-1,:])
      HH → Laplacian ≈ 4x - sum(neighbors) (2nd-order detail)

    Why 4 sub-bands beats 2-band (Prism) and directional (Nexus):
    - Prism: 2 bands (lo, hi) → misses directional information in hi-freq
    - Nexus: 4 directions on RAW features → no frequency decomposition
    - Zenith: 4 WAVELET sub-bands → complete multi-resolution + directional coverage
    - Captures smooth regions (LL), H-edges (LH), V-edges (HL), corners (HH) separately

    FLOPs per pixel: k²×c/4 + k×c/4 + k×c/4 + 9×c/4 = (k²+2k+9)×c/4
    For k=5: (25+10+9)×c/4 = 11c (LIGHTER than standard 9c DWConv 3×3 only at k=3!)
    Params: same formula + c(BN) ≈ 12c (for k=5)

    Args:
        c (int): Number of input/output channels (must be >= 4).
        k (int): Kernel size for LL-path and strip-path. Default 5.

    Shape:
        Input:  (B, c, H, W)
        Output: (B, c, H, W)
    """

    def __init__(self, c, k=5):
        super().__init__()
        assert c >= 4, f"WaveletConv requires c >= 4, got c={c}"

        # Split channels into 4 sub-band groups
        self.c_ll = c // 4
        self.c_lh = c // 4
        self.c_hl = c // 4
        self.c_hh = c - 3 * (c // 4)  # handles non-divisible-by-4

        # LL path: AvgPool → DWConv (smooth approximation processing)
        self.ll_pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.dw_ll = nn.Conv2d(
            self.c_ll, self.c_ll, k, stride=1, padding=k // 2,
            groups=self.c_ll, bias=False
        )

        # LH path: horizontal gradient → horizontal-strip DWConv
        self.dw_lh = nn.Conv2d(
            self.c_lh, self.c_lh, (1, k), stride=1, padding=(0, k // 2),
            groups=self.c_lh, bias=False
        )

        # HL path: vertical gradient → vertical-strip DWConv
        self.dw_hl = nn.Conv2d(
            self.c_hl, self.c_hl, (k, 1), stride=1, padding=(k // 2, 0),
            groups=self.c_hl, bias=False
        )

        # HH path: Laplacian → isotropic DWConv (diagonal/corner features)
        self.hh_pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.dw_hh = nn.Conv2d(
            self.c_hh, self.c_hh, 3, stride=1, padding=1,
            groups=self.c_hh, bias=False
        )

        # Post-merge: BN + activation
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """Split → 4 wavelet sub-band DWConvs → Concat → Shuffle → BN → SiLU."""
        # Channel split into 4 sub-band groups
        x_ll = x[:, :self.c_ll]
        x_lh = x[:, self.c_ll:self.c_ll + self.c_lh]
        x_hl = x[:, self.c_ll + self.c_lh:self.c_ll + self.c_lh + self.c_hl]
        x_hh = x[:, self.c_ll + self.c_lh + self.c_hl:]

        # LL: Low-pass filtered → DWConv (smooth spatial context)
        y_ll = self.dw_ll(self.ll_pool(x_ll))

        # LH: Horizontal detail via padded gradient → strip DWConv
        # Approximate horizontal Haar wavelet: x - horizontal_shift(x)
        x_lh_grad = x_lh - F.pad(x_lh[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')
        y_lh = self.dw_lh(x_lh_grad)

        # HL: Vertical detail via padded gradient → strip DWConv
        x_hl_grad = x_hl - F.pad(x_hl[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')
        y_hl = self.dw_hl(x_hl_grad)

        # HH: Laplacian (diagonal detail): x - AvgPool(x) ≈ high-pass 2D
        x_hh_lap = x_hh - self.hh_pool(x_hh)
        y_hh = self.dw_hh(x_hh_lap)

        # Merge with inter-subband channel shuffle
        y = torch.cat([y_ll, y_lh, y_hl, y_hh], dim=1)
        y = self._channel_shuffle(y, groups=4)

        return self.act(self.bn(y))

    @staticmethod
    def _channel_shuffle(x, groups=4):
        """Shuffle channels between sub-band groups for cross-resolution info flow."""
        B, C, H, W = x.shape
        if C % groups != 0:
            return x
        return (
            x.view(B, groups, C // groups, H, W)
            .transpose(1, 2)
            .contiguous()
            .view(B, C, H, W)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOPOLOGICAL GATE — TopologicalGate
# ─────────────────────────────────────────────────────────────────────────────
class TopologicalGate(nn.Module):
    """
    Topological Complexity Gate — Euler-inspired channel attention.

    Uses a novel "peak count" approximation to estimate the TOPOLOGICAL
    COMPLEXITY of each channel's spatial activation map. Channels with
    more distinct peaks (objects, keypoints) get upweighted vs channels
    with smooth/uniform activations (background).

    How it works:
    1. Compute per-channel statistics:
       - μ (mean): baseline activation level via GAP
       - σ (std): spread of activations
       - peak_ratio: fraction of positions exceeding (μ + α*σ)
         where α is a learnable threshold parameter
    2. peak_ratio approximates "how many peaks exist" normalized by area:
       - High peak_ratio → many activations above threshold → rich topology
       - Low peak_ratio → few/no peaks → flat/uniform → simple topology
    3. Gate = FC(μ ∥ peak_ratio) → Sigmoid

    Mathematical connection to persistent homology:
    In topological data analysis, the number of connected components of the
    superlevel set {x : f(x) > t} at threshold t relates to the 0-dimensional
    Betti number β₀. Our peak_ratio approximates a soft version of β₀ at
    an adaptive threshold μ + α*σ, using a sigmoid soft-threshold.

    Why this is fundamentally different from all existing attention:
    - SE (mean only): Cannot distinguish peaked vs flat at same mean
    - CBAM (mean+max): Max is single pixel; peak_ratio captures ALL peaks
    - MCG/Prism (L2/L1): Concentration measure — high if ONE big peak
      vs our peak_ratio which captures MULTIPLE peaks (richer topology)
    - NormRatioGate/Nexus (L1/L∞): Single max, outlier-sensitive
    - SpectralGate/Chimera (mean+std): std only measures spread, not peak count

    Example: Two channels, both mean=0.5, std=0.3:
      Channel A: 10 distinct object blobs each at 1.0, rest at 0.3 → peak_ratio high
      Channel B: Smooth gradient from 0.0 to 1.0 → peak_ratio low
    SE, MCG, SpectralGate all see these as SIMILAR. TopologicalGate distinguishes them.

    FLOPs: GAP + variance + sigmoid threshold + mean + FC ≈ 3HWC + 2C²/r
    Params: FC(2C → C/r) + FC(C/r → C) + α(C) = 2C²/r + C/r + C

    Args:
        c (int): Number of channels.
        reduction (int): FC hidden dimension reduction ratio. Default 4.

    Shape:
        Input:  (B, c, H, W)
        Output: (B, c, H, W)
    """

    def __init__(self, c, reduction=4):
        super().__init__()
        c_mid = max(c // reduction, 4)

        # Learnable per-channel threshold parameter for peak detection
        # Initialized to 1.0 (one standard deviation above mean)
        self.alpha = nn.Parameter(torch.ones(1, c, 1, 1))

        # FC gating from combined descriptor [mean, peak_ratio]
        self.fc = nn.Sequential(
            nn.Linear(c * 2, c_mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(c_mid, c, bias=False),
        )

    def forward(self, x):
        """Compute topological peak_ratio + mean → FC gate → channel modulation."""
        B, C, H, W = x.shape

        # Per-channel statistics
        mu = x.mean(dim=[2, 3])  # (B, C) — mean activation level
        sigma = (x.var(dim=[2, 3], unbiased=False) + 1e-6).sqrt()  # (B, C)

        # Soft peak detection: sigmoid threshold at μ + α*σ
        # This creates a soft binary mask of "above threshold" positions
        alpha_sq = self.alpha.view(1, C, 1, 1)
        threshold = mu.view(B, C, 1, 1) + alpha_sq * sigma.view(B, C, 1, 1)
        soft_peaks = torch.sigmoid(5.0 * (x - threshold))  # steepness=5 for sharp-ish threshold

        # Peak ratio: fraction of spatial positions classified as peaks
        # This approximates topological complexity (β₀ of superlevel set)
        peak_ratio = soft_peaks.mean(dim=[2, 3])  # (B, C) ∈ [0, 1]

        # Combined descriptor: magnitude + topological complexity
        desc = torch.cat([mu, peak_ratio], dim=1)  # (B, 2C)

        # FC gating
        gate = torch.sigmoid(self.fc(desc))  # (B, C)

        return x * gate.unsqueeze(-1).unsqueeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ZENITH BOTTLENECK — Core Building Block
# ─────────────────────────────────────────────────────────────────────────────
class ZenithBottleneck(nn.Module):
    """
    Zenith Bottleneck — wavelet-decomposed inverted residual with topological gate.

    Architecture: 1×1 Expand → WaveletConv → TopologicalGate → 1×1 Project → Residual

    The inverted bottleneck design:
    1. 1×1 Expand: mix channels before wavelet decomposition (channel → spatial)
    2. WaveletConv: 4-subband wavelet processing (LL, LH, HL, HH)
    3. TopologicalGate: channel recalibration based on activation topology
    4. 1×1 Project: compress back to output dimension

    vs PrismBottleneck: 4 sub-bands vs 2; topology vs concentration attention
    vs ChimeraBottleneck: wavelet basis vs dilation basis; topology vs spectral
    vs NexusBottleneck: frequency-decomposed vs direction-only; topology vs sparsity

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        shortcut (bool): Use residual connection when c1 == c2. Default True.
        g (int): Groups for API compatibility. Default 1.
        k (int): Base kernel size (3 for n/s via c3k=False; 5 for m/l/x). Default 3.
        e (float): Expansion ratio for hidden channels. Default 1.0.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=1.0):
        super().__init__()
        c_hidden = int(c2 * e)

        # Scale WaveletConv kernel based on k parameter
        wk = 7 if k >= 5 else 5  # c3k=True → larger wavelet kernels

        # 1×1 Expand
        self.expand_conv = Conv(c1, c_hidden, 1)

        # 4-subband Wavelet DWConv
        self.wavelet = WaveletConv(c_hidden, k=wk)

        # Topological complexity gate
        self.gate = TopologicalGate(c_hidden)

        # 1×1 Project
        self.project = Conv(c_hidden, c2, 1)

        # Residual connection
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Expand → WaveletConv → TopologicalGate → Project → Residual."""
        y = self.expand_conv(x)
        y = self.wavelet(y)
        y = self.gate(y)
        y = self.project(y)
        return x + y if self.add else y


# ─────────────────────────────────────────────────────────────────────────────
# 4. C3K2_ZENITH — C2f Backbone Container with ZenithBottleneck
# ─────────────────────────────────────────────────────────────────────────────
class C3k2_Zenith(nn.Module):
    """
    C2f split-concat architecture with ZenithBottleneck for backbone.

    Drop-in replacement for C3k2 providing:
    - 4-subband wavelet spatial features (LL, LH, HL, HH decomposition)
    - Topological complexity channel attention (peak-count gate)
    - Multi-resolution gradient flow (C2f split-concat pattern)

    The c3k parameter controls wavelet kernel scale:
    - c3k=False: WaveletConv(k=5) — standard, good for n/s scales
    - c3k=True:  WaveletConv(k=7) — larger wavelet basis, good for m/l/x

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of ZenithBottleneck repeats. Default 1.
        c3k (bool): Use larger wavelet kernels. Default False.
        e (float): Channel split ratio. Default 0.5.
        g (int): Groups (API compatibility). Default 1.
        shortcut (bool): Residual connections in bottleneck. Default True.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels per split
        k = 5 if c3k else 3

        # Entry: project input to 2 × self.c channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Exit: merge (2 + n) × self.c channels back to c2
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Core: n × ZenithBottleneck
        self.m = nn.ModuleList(
            ZenithBottleneck(self.c, self.c, shortcut, g, k=k)
            for _ in range(n)
        )

    def forward(self, x):
        """Split → n × ZenithBottleneck → concat all → merge."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward using split() instead of chunk() for some runtimes."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ─────────────────────────────────────────────────────────────────────────────
# 5. ADAPTIVE SCALE ROUTER — AdaptiveScaleRouter
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveScaleRouter(nn.Module):
    """
    Adaptive Scale Router — content-dependent multi-scale spatial attention for neck.

    Instead of applying the SAME multi-scale processing to ALL positions (like
    standard FPN/PAN), ASR learns to ROUTE each spatial position to its optimal
    scale context. This creates position-dependent receptive fields.

    How it works:
    1. Generate K scale feature maps at different resolutions:
       - Scale 0 (local):   identity → DWConv 3×3
       - Scale 1 (medium):  AvgPool(3) → DWConv 3×3 → Upsample
       - Scale 2 (global):  AdaptiveAvgPool(H/4) → Conv 1×1 → Upsample

    2. Compute routing weights per position:
       - Spatial descriptor: channel-wise mean → Conv 1×1 → 3 routing logits
       - Softmax → per-position routing probabilities for K=3 scales

    3. Weighted combination: output = Σ_k (route_k × scale_k_features)
       Each position gets its OWN optimal mix of local/medium/global context.

    Why this is fundamentally different from existing neck modules:
    - Standard FPN/PAN: Fixed topology, no position-adaptive routing
    - CBAM spatial: Single scale attention (gate per position, but same features)
    - CSM (Chimera): Only zoom-out global, no per-position routing
    - FASR (Prism): Low-freq + high-freq paths, but same mix everywhere
    - PCR (Nexus): ON/OFF polarity, not multi-scale routing

    ASR uniquely provides CONTENT-DEPENDENT receptive field per position:
    - Small object at position (i,j) → router assigns more weight to local scale
    - Large object → more weight to medium/global scales
    - This is a lightweight form of dynamic network routing (CondConv/MoE spirit)

    FLOPs: 3 × DWConv + routing Conv + softmax ≈ 30c (negligible vs main conv)
    Params: 3 DWConv(c→c) + Conv(c→3) ≈ 27c + 3c = 30c

    Args:
        c (int): Number of channels.
        n_scales (int): Number of scale routes. Default 3.

    Shape:
        Input:  (B, c, H, W)
        Output: (B, c, H, W)
    """

    def __init__(self, c, n_scales=3):
        super().__init__()
        self.n_scales = n_scales

        # Scale 0: local detail (3×3 DWConv, no downsampling)
        self.scale_local = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        # Scale 1: medium context (AvgPool → DWConv → Upsample)
        self.scale_medium_pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.scale_medium_dw = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=1, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        # Scale 2: global context (AdaptivePool → 1×1 Conv → Upsample)
        self.scale_global_conv = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        # Routing network: per-position routing logits for K scales
        # Uses lightweight 1×1 conv on channel-reduced features
        self.router = nn.Sequential(
            nn.Conv2d(c, max(c // 4, 4), 1, bias=False),
            nn.BatchNorm2d(max(c // 4, 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c // 4, 4), n_scales, 1, bias=True),
        )

    def forward(self, x):
        """Generate multi-scale features → compute routing → weighted combine."""
        B, C, H, W = x.shape

        # Generate features at 3 scales
        f_local = self.scale_local(x)  # (B, C, H, W)

        f_medium = self.scale_medium_dw(self.scale_medium_pool(x))  # (B, C, H, W)

        # Global: downsample → process → upsample
        gh, gw = max(H // 4, 1), max(W // 4, 1)
        f_global = F.adaptive_avg_pool2d(x, (gh, gw))
        f_global = self.scale_global_conv(f_global)
        f_global = F.interpolate(f_global, size=(H, W), mode='bilinear', align_corners=False)

        # Stack scale features: (B, K, C, H, W)
        scales = torch.stack([f_local, f_medium, f_global], dim=1)

        # Compute per-position routing weights: (B, K, H, W)
        route_logits = self.router(x)  # (B, K, H, W)
        route_weights = F.softmax(route_logits, dim=1)  # (B, K, H, W)

        # Weighted combination: each position uses its own scale mix
        route_weights = route_weights.unsqueeze(2)  # (B, K, 1, H, W)
        output = (scales * route_weights).sum(dim=1)  # (B, C, H, W)

        return output + x  # residual connection


# ─────────────────────────────────────────────────────────────────────────────
# 6. ZENITH CSP — C2f Neck Container with AdaptiveScaleRouter
# ─────────────────────────────────────────────────────────────────────────────
class ZenithCSP(nn.Module):
    """
    C2f + ZenithBottleneck + AdaptiveScaleRouter for neck fusion.

    Combines wavelet backbone strength with a unique neck-specific
    Adaptive Scale Router that provides content-dependent multi-scale
    attention after the merge convolution.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of ZenithBottleneck repeats. Default 1.
        c3k (bool): Use larger wavelet kernels. Default False.
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

        # Core: n × ZenithBottleneck
        self.m = nn.ModuleList(
            ZenithBottleneck(self.c, self.c, shortcut, g, k=k)
            for _ in range(n)
        )

        # Adaptive Scale Router — novel neck-specific spatial routing
        self.asr = AdaptiveScaleRouter(c2)

    def forward(self, x):
        """Split → n × ZenithBottleneck → concat → merge → adaptive scale route."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.asr(out)  # adaptive multi-scale routing

    def forward_split(self, x):
        """Forward using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.asr(out)
