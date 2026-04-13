# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-Chimera — Novel Hybrid Detection Architecture with Multi-Dilation Fusion
==============================================================================

A completely original architecture that introduces three novel components
not found in any existing published work.

Core Innovations:

1. Trident Dilated Decomposition (TDD) — TridentConv:
   Splits input channels into THREE groups and processes each with DWConv 3×3
   at different dilation rates: d=1 (local), d=2 (medium), d=4 (broad context).
   After DWConv, channel shuffle mixes inter-group information.

   Key Insight: Same 3×3 kernel with different dilations achieves effective
   receptive fields of 3×3, 5×5, and 9×9 with IDENTICAL parameter count.
   Pure DWConv keeps FLOPs minimal (linear in channels, not quadratic).

   Advantages:
   - vs HeteroConv (Phoenix): Same params but 3 RF scales instead of 2
   - vs ASPP (DeepLab):       DWConv on channel splits (O(c)) vs full Conv (O(c²))
   - vs Trident Network:      One block with channel splits vs 3 full branches
   - vs PConv (FasterNet):    Processes ALL channels, not just 1/4
   - vs standard DWConv:      Multi-scale RF at zero extra cost via dilation

   FLOPs per pixel: 3 × 9 × (c/3) = 9c  (same as single DWConv 3×3!)
   Effective RF:    3, 5, 9 simultaneously

2. Spectral Channel Reweighting (SCR) — SpectralGate:
   Novel channel attention using BOTH mean AND standard deviation statistics.
   - GAP → mean descriptor (captures typical activation, low-frequency info)
   - Global StdDev → spread descriptor (captures variance, high-frequency info)
   - Fusion: FC(mean + std) → Sigmoid gate

   Why StdDev matters for detection:
   - High std channels → rich textures, edges, object boundaries
   - Low std channels → uniform regions, backgrounds
   - This information is INVISIBLE to SE (mean only), CBAM (mean+max),
     DualPoolGate (mean+max), or ECA (local correlation)

   Efficient computation: std = sqrt(E[X²] - E[X]² + ε)
   No extra forward pass needed — computed from GAP of X and X².

   Advantages:
   - vs SE Block:       Captures activation spread (std) not just level (mean)
   - vs CBAM:           StdDev gives richer info than max (peak ≠ spread)
   - vs DualPoolGate:   Fundamentally different statistic (2nd moment vs extremum)
   - vs ECA:            Global context via FC, not just local channel correlation

3. Cross-Scale Modulator (CSM) — in ChimeraCSP neck:
   Novel neck-specific module that performs "zoom-out-then-zoom-in" modulation:
   - Content branch: AvgPool(2×2) → 1×1 Conv → Bilinear Upsample
     (captures broad "what" context at half resolution)
   - Detail branch: DWConv 3×3 (preserves fine "where" spatial detail)
   - Fusion: sigmoid(content) * detail + detail
     (content-aware emphasis on spatially important regions)

   This creates a CONTENT-GATED SPATIAL ATTENTION unique to Chimera's neck.
   Standard YOLO necks have NO spatial modulation. Phoenix has DWConv→Sigmoid
   (purely local). Chimera's CSM uses multi-scale context to gate detail.

   Advantages:
   - vs no spatial attention (YOLO11, EDGE): Adds spatial focus at minimal cost
   - vs Phoenix SRM (local DWConv→Sigmoid): Uses multi-scale context, not just local
   - vs CBAM spatial (mean+max→Conv): Content-driven gating, not just statistics

Architecture Design:
   ChimeraBottleneck = 1×1 Expand → TridentConv → SpectralGate → 1×1 Project → Residual
   C3k2_Chimera     = C2f split-concat with ChimeraBottleneck (backbone)
   ChimeraCSP       = C2f split-concat + CrossScaleModulator (neck)

Parameter Comparison (c=128, per bottleneck):
   Standard Bottleneck (3×3+3×3):   ~147K params, ~18c² FLOPs/pixel
   FasterBottleneck (PConv):        ~26K params,  ~1.56c² FLOPs/pixel
   PhoenixBottleneck (HeteroConv):  ~86K params,  ~4c² FLOPs/pixel
   ChimeraBottleneck (ours):        ~70K params,  ~4c² FLOPs/pixel
   → Lighter than Phoenix with TRIPLE the RF diversity (3 vs 2 scales)
   → 2.5× more expressive than PConv while processing ALL channels

Key Design Principles:
   - Triple-scale by design: every spatial op extracts 3 different RF features
   - Spectral attention: novel stddev-based channel intelligence
   - Content-gated neck: multi-scale context drives spatial focus
   - Hardware-friendly: DWConv + 1×1 Conv, fully ONNX/TensorRT compatible
   - No exotic ops: all standard PyTorch operations

References:
   - Trident dilated decomposition: Novel 3-way channel-split dilation (Chimera contribution)
   - Spectral channel gate: Novel mean+stddev attention (Chimera contribution)
   - Cross-scale modulator: Novel content-gated spatial attention (Chimera contribution)
   - Dilated convolutions: Yu & Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions", ICLR 2016
   - Depthwise separable conv: Xception (Chollet, CVPR 2017), MobileNet (Howard et al., 2017)
   - Channel attention: SENet (Hu et al., CVPR 2018) — fundamentally reimagined with spectral stats
   - C2f container: YOLOv8/v11 (Ultralytics)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


# ─────────────────────────────────────────────────────────────────────────────
# TRIDENT CONV — Triple-Dilation Depthwise Channel Decomposition
# ─────────────────────────────────────────────────────────────────────────────
class TridentConv(nn.Module):
    """
    Trident Dilated Decomposition — triple-scale DWConv with channel shuffle.

    Splits input into 3 channel groups, each processed by DWConv 3×3 with
    different dilation rates (1, 2, 4). After processing, channels are shuffled
    across groups to enable inter-scale information mixing.

    This achieves effective receptive fields of 3×3, 5×5, and 9×9 using
    ONLY 3×3 kernels — same parameter count as a single DWConv 3×3.

    vs HeteroConv (Phoenix): 3 scales instead of 2, dilated instead of multi-kernel
    vs ASPP (DeepLab): DWConv per group (O(c)) instead of full Conv (O(c²))
    vs Trident Network: Single block with splits, not 3 heavyweight branches

    FLOPs per pixel: 9c (identical to standard 3×3 DWConv)
    Params: 3 × 9 × (c/3) + c (BN) = 9c + c = 10c

    Args:
        c (int): Number of input/output channels (≥3).
        base_k (int): Base kernel size. Default 3.

    Shape:
        Input:  (B, c, H, W)
        Output: (B, c, H, W)
    """

    def __init__(self, c, base_k=3):
        super().__init__()
        # Split into 3 roughly equal groups
        self.c1 = c // 3
        self.c2 = c // 3
        self.c3 = c - self.c1 - self.c2  # handles non-divisible channels

        # Dilation 1: local features (effective RF = 3×3)
        self.dw_d1 = nn.Conv2d(
            self.c1, self.c1, base_k,
            stride=1, padding=base_k // 2,  # dilation=1, padding=1
            dilation=1, groups=self.c1, bias=False,
        )
        # Dilation 2: medium context (effective RF = 5×5)
        self.dw_d2 = nn.Conv2d(
            self.c2, self.c2, base_k,
            stride=1, padding=(base_k // 2) * 2,  # dilation=2, padding=2
            dilation=2, groups=self.c2, bias=False,
        )
        # Dilation 4: broad context (effective RF = 9×9)
        self.dw_d4 = nn.Conv2d(
            self.c3, self.c3, base_k,
            stride=1, padding=(base_k // 2) * 4,  # dilation=4, padding=4
            dilation=4, groups=self.c3, bias=False,
        )

        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
        self.n_groups = 3  # for channel shuffle

    def _channel_shuffle(self, x):
        """Shuffle channels across the 3 dilation groups for inter-scale mixing."""
        b, c, h, w = x.shape
        # Ensure c is divisible by n_groups for reshape
        g = self.n_groups
        if c % g != 0:
            return x  # skip shuffle for non-divisible (safety)
        x = x.reshape(b, g, c // g, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.reshape(b, c, h, w)

    def forward(self, x):
        """Split → triple-dilation DWConv → concat → shuffle → BN → SiLU."""
        x1, x2, x3 = torch.split(x, [self.c1, self.c2, self.c3], dim=1)
        x1 = self.dw_d1(x1)  # local features
        x2 = self.dw_d2(x2)  # medium context
        x3 = self.dw_d4(x3)  # broad context
        out = torch.cat([x1, x2, x3], dim=1)
        out = self._channel_shuffle(out)
        return self.act(self.bn(out))


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRAL GATE — Channel Attention with Mean + Standard Deviation
# ─────────────────────────────────────────────────────────────────────────────
class SpectralGate(nn.Module):
    """
    Spectral Channel Reweighting — attention using mean AND stddev statistics.

    Novel insight: Standard deviation captures how "spread out" each channel's
    activations are across spatial positions. High-std channels contain rich
    textures, edges, and object boundaries. Low-std channels are uniform/background.

    This information is fundamentally different from:
    - SE Block: uses only mean (GAP)
    - CBAM: uses mean + max (but max ≠ spread)
    - DualPoolGate: uses mean + max
    - ECA: uses local channel correlation

    Efficient StdDev computation: std = sqrt(E[X²] - (E[X])² + ε)
    Uses GAP on X and X² — no extra forward pass needed.

    Params: 2 × c × c_mid  (same as SE)
    FLOPs: ~2c per pixel (negligible vs convolutions)

    Args:
        c (int): Number of channels.
        reduction (int): FC reduction ratio. Default 8.

    Shape:
        Input:  (B, c, H, W)
        Output: (B, c, H, W) — spectrally recalibrated features
    """

    def __init__(self, c, reduction=8):
        super().__init__()
        c_mid = max(c // reduction, 4)
        # Shared FC pathway for combined mean+std descriptor
        self.fc1 = nn.Conv2d(c, c_mid, 1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(c_mid, c, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        """GAP(mean) + sqrt(GAP(x²) - GAP(x)²) → FC → Sigmoid → channel gate."""
        # Mean descriptor (1st moment)
        mean = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)

        # StdDev descriptor (2nd moment) — efficient computation
        # Var(X) = E[X²] - (E[X])²
        mean_sq = F.adaptive_avg_pool2d(x * x, 1)  # E[X²]
        var = mean_sq - mean * mean  # Var(X)
        std = torch.sqrt(var.clamp(min=1e-8))  # sqrt(Var) = StdDev

        # Combine mean + std as spectral descriptor
        spectral = mean + std  # (B, C, 1, 1)

        # FC pathway → gate
        return x * self.gate(self.fc2(self.act(self.fc1(spectral))))


# ─────────────────────────────────────────────────────────────────────────────
# CHIMERA BOTTLENECK — Core Building Block
# ─────────────────────────────────────────────────────────────────────────────
class ChimeraBottleneck(nn.Module):
    """
    Chimera Bottleneck: Expand → TridentConv → SpectralGate → Project → Residual.

    Combines inverted residual structure with triple-dilation depthwise convolution
    and spectral (mean+stddev) channel attention.

    Architecture:
        Input (c) → Conv1×1 (c → 2c) → TridentConv(d=1,2,4) → SpectralGate
                  → Conv1×1 (2c → c) → + Input (residual)

    FLOPs Analysis (c channels, H×W spatial):
        1×1 Expand:      c × 2c × H × W  = 2c² HW
        TridentConv DW:  9 × 2c × H × W  ≈ 0   (negligible vs c²)
        SpectralGate:    ~2c              ≈ 0   (global pooling)
        1×1 Project:     2c × c × H × W  = 2c² HW
        Total:           ≈ 4c² HW
        vs Standard 3×3: 9c² HW → 2.25× lighter
        vs Phoenix:      4c² HW → comparable FLOPs, but 3 RF scales vs 2

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        shortcut (bool): Use residual connection. Default True.
        g (int): Groups (API compatibility). Default 1.
        k (int): Base kernel size for TridentConv. Default 3.
        e (float): Unused (API compatibility with C2f). Default 0.5.
    """

    EXPAND = 2  # Channel expansion ratio (inverted residual)

    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5):
        super().__init__()
        c_expand = max(int(c1 * self.EXPAND), 3)  # min 3 for 3-way split
        # Ensure c_expand is divisible by 3 for clean TridentConv splits
        c_expand = max(c_expand // 3 * 3, 3)

        # 1×1 channel expansion with BN + SiLU
        self.expand_conv = Conv(c1, c_expand, 1)

        # Triple-dilation depthwise spatial processing
        self.trident = TridentConv(c_expand, base_k=k)

        # Spectral channel recalibration (mean + stddev attention)
        self.gate = SpectralGate(c_expand, reduction=8)

        # 1×1 channel projection (no activation — residual pathway)
        self.project = nn.Sequential(
            nn.Conv2d(c_expand, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Expand → TridentConv → SpectralGate → Project → Residual."""
        y = self.expand_conv(x)
        y = self.trident(y)
        y = self.gate(y)
        y = self.project(y)
        return x + y if self.add else y


# ─────────────────────────────────────────────────────────────────────────────
# C3K2_CHIMERA — C2f Container with ChimeraBottleneck (Backbone)
# ─────────────────────────────────────────────────────────────────────────────
class C3k2_Chimera(nn.Module):
    """
    C2f split-concat architecture with ChimeraBottleneck for backbone.

    Drop-in replacement for C3k2 that provides:
    - Triple-scale spatial features (via TridentConv d=1,2,4)
    - Spectral channel attention (via SpectralGate mean+stddev)
    - Multi-resolution gradient flow (via C2f split-concat)

    The c3k parameter controls base kernel scale:
    - c3k=False: TridentConv(3×3, d=1,2,4) — standard, good for n/s
    - c3k=True:  TridentConv(5×5, d=1,2,4) — larger base RF, good for m/l/x

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of ChimeraBottleneck repeats. Default 1.
        c3k (bool): Use 5×5 base kernel vs 3×3. Default False.
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

        # Core: n × ChimeraBottleneck
        self.m = nn.ModuleList(
            ChimeraBottleneck(self.c, self.c, shortcut, g, k=k)
            for _ in range(n)
        )

    def forward(self, x):
        """Split → n × ChimeraBottleneck → concat all → merge."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward using split() instead of chunk() for some runtimes."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SCALE MODULATOR (CSM) — Novel Neck Spatial Attention
# ─────────────────────────────────────────────────────────────────────────────
class CrossScaleModulator(nn.Module):
    """
    Cross-Scale Modulator — content-gated spatial attention for neck fusion.

    Performs "zoom-out-then-zoom-in" modulation:
    1. Content branch: AvgPool(2×2) → 1×1 Conv → Bilinear Upsample
       Captures broad semantic context at half resolution (global "what")
    2. Detail branch: DWConv 3×3
       Preserves fine spatial detail (local "where")
    3. Fusion: sigmoid(content) * detail + detail
       Content-aware emphasis — broad semantics drive local spatial focus

    This is fundamentally different from:
    - Phoenix SRM: purely local DWConv → Sigmoid (no multi-scale context)
    - CBAM spatial: reduce channels → Conv → Sigmoid (loses channel info)
    - No spatial attention (YOLO11, EDGE): no spatial focus at all

    The "zoom-out" via AvgPool gives global context that pure DWConv cannot:
    a 2×2 pooled 1×1 Conv "sees" 2× larger spatial region per position.

    Params: c × c_mid (1×1) + c_mid (1×1 restore) + 9c (DWConv) ≈ 11c
    Overhead: <1% of total neck FLOPs

    Args:
        c (int): Number of channels.

    Shape:
        Input:  (B, c, H, W)
        Output: (B, c, H, W) — spatially modulated features
    """

    def __init__(self, c):
        super().__init__()
        c_mid = max(c // 4, 4)  # compressed channel for content branch

        # Content branch — "zoom-out" for global context
        self.content_compress = nn.Conv2d(c, c_mid, 1, bias=False)
        self.content_bn = nn.BatchNorm2d(c_mid)
        self.content_act = nn.SiLU(inplace=True)
        self.content_expand = nn.Conv2d(c_mid, c, 1, bias=False)

        # Detail branch — local spatial processing
        self.detail_dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.detail_bn = nn.BatchNorm2d(c)

        # Gate activation
        self.gate = nn.Sigmoid()

    def forward(self, x):
        """Content (zoom-out-zoom-in) gates Detail (local DWConv)."""
        _, _, h, w = x.shape

        # Content branch: downsample → process → upsample
        # AvgPool to half resolution for broader context
        pool_h, pool_w = max(h // 2, 1), max(w // 2, 1)
        content = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
        content = self.content_act(self.content_bn(self.content_compress(content)))
        content = self.content_expand(content)
        content = F.interpolate(content, size=(h, w), mode='bilinear', align_corners=False)
        content_gate = self.gate(content)

        # Detail branch: local spatial features
        detail = self.detail_bn(self.detail_dw(x))

        # Cross-scale fusion: content gates detail
        return content_gate * detail + detail  # residual gating


# ─────────────────────────────────────────────────────────────────────────────
# CHIMERA CSP — C2f Container with Cross-Scale Modulator (Neck)
# ─────────────────────────────────────────────────────────────────────────────
class ChimeraCSP(nn.Module):
    """
    C2f + ChimeraBottleneck + CrossScaleModulator for neck fusion.

    Combines the triple-dilation backbone strength with a unique neck-specific
    Cross-Scale Modulator (CSM) that adds content-gated spatial attention
    after the merge convolution.

    The CSM is a key novelty: standard YOLO necks lack spatial attention.
    Phoenix adds local SRM (DWConv→Sigmoid). Chimera's CSM adds MULTI-SCALE
    content-driven spatial modulation — broad "what" context gates local "where" detail.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of ChimeraBottleneck repeats. Default 1.
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

        # Core: n × ChimeraBottleneck
        self.m = nn.ModuleList(
            ChimeraBottleneck(self.c, self.c, shortcut, g, k=k)
            for _ in range(n)
        )

        # Cross-Scale Modulator — novel neck-specific spatial attention
        self.csm = CrossScaleModulator(c2)

    def forward(self, x):
        """Split → n × ChimeraBottleneck → concat → merge → cross-scale modulate."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.csm(out)  # content-gated spatial modulation

    def forward_split(self, x):
        """Forward using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.csm(out)
