# YOLO-Prism V2: Tri-Band Frequency-Decomposed Detection Architecture
## Evolution of YOLO-Prism with Multi-Band Signal Processing

---

## 1. MOTIVATION — Why Prism V2?

### 1.1 Prism V1 Results
YOLO-Prism V1 achieved excellent results on BCCD dataset:
- Significantly fewer parameters than baseline YOLOv11
- Lower GFLOPs while maintaining detection accuracy
- Novel frequency-decomposed approach proved effective

### 1.2 Identified V1 Limitations

| # | Limitation | Impact | V2 Solution |
|---|-----------|--------|-------------|
| 1 | Only 2 frequency bands (LF+HF) | Missing mid-frequency textures | **TriFreqConv** with 3 bands |
| 2 | Fixed frequency cutoff | Same cutoff at all pyramid levels | Adaptive filter params per stage |
| 3 | MCG only uses L1+γ | Cannot distinguish textured vs flat | **FreqContrastGate** adds σ² |
| 4 | FASR has fixed LF:HF ratio | P3 and P5 get same emphasis | **AdaptiveFreqRefine** with learnable α |
| 5 | DualFreqConv uses c/2 split | 50% overhead per band | c/3 split → 16% fewer FLOPs |

### 1.3 Key Insight: Mid-Frequency Band

The critical observation from signal processing theory:

```
Frequency Spectrum of Visual Features:
├── Low Frequency (LF):  smooth regions, backgrounds, semantic context
├── Mid Frequency (MF):  textures, patterns, surface detail ← MISSING IN V1
└── High Frequency (HF): edges, boundaries, fine detail

Object Detection needs ALL THREE:
- Classification: relies on TEXTURES (MF) — cat vs dog = fur pattern
- Localization: relies on EDGES (HF) — bounding box = sharp transitions
- Context: relies on SEMANTICS (LF) — sky above, road below
```

V1 forces texture features into both LF and HF paths, diluting the representation.
V2 isolates textures with a dedicated band-pass path.

---

## 2. ARCHITECTURE INNOVATIONS

### 2.1 TriFreqConv — Tri-Band Frequency Decomposed Convolution

```
Input (B, C, H, W) — split into 3 groups (C/3 each)
│
├── Group 1 (LF): DWConv K_lo×K_lo on raw features
│     → Captures smooth spatial context
│
├── Group 2 (MF): BandPass(AvgPool_fine - AvgPool_coarse) → DWConv 3×3  [NEW]
│     → Isolates mid-frequency textures/patterns
│     → BandPass is PARAMETER-FREE (2 AvgPool ops)
│
├── Group 3 (HF): HighPass(x - AvgPool_fine) → DWConv K_hi×K_hi
│     → Captures edge/detail content
│
└── Concat → Channel Shuffle (3 groups) → BN → SiLU
```

**Band-Pass Filter Theory:**
```
AvgPool_fine  = Low-pass with cutoff f_fine   (passes LF + MF)
AvgPool_coarse = Low-pass with cutoff f_coarse (passes LF only)

BandPass = AvgPool_fine - AvgPool_coarse = MF band only!

Transfer functions:
  H_LP_fine(ω)   = sinc(k_fine·ω/2)
  H_LP_coarse(ω) = sinc(k_coarse·ω/2)
  H_BP(ω)        = H_LP_fine(ω) - H_LP_coarse(ω)  ← mid-frequency pass
  H_HP(ω)        = 1 - H_LP_fine(ω)                ← high-frequency pass
```

**FLOPs Comparison:**
| Component | DualFreqConv (V1) | TriFreqConv (V2) | Savings |
|-----------|-------------------|-------------------|---------|
| DWConv (k=5,3) | (25+9)×c/2 = 17c | (25+9+9)×c/3 = 14.3c | **-16%** |
| DWConv (k=7,5) | (49+25)×c/2 = 37c | (49+9+25)×c/3 = 27.7c | **-25%** |

### 2.2 FreqContrastGate — Tri-Moment Channel Attention

```
Input (B, C, H, W)
│
├── L1 = mean(|X|)          → magnitude (how strong)
├── L2 = sqrt(mean(X²))     → RMS magnitude
├── γ = L2/L1               → concentration ratio [Cauchy-Schwarz]
├── σ² = E[X²] - E[X]²     → frequency variance [NEW, near-zero cost]
│
├── desc = L1 + β·γ + λ·σ²  (β, λ learnable per-channel)
├── FC(desc) → Sigmoid → gate
│
└── Output = X × gate
```

**Why σ² is almost free:**
- `E[X²]` is ALREADY computed for L2 (just `x.pow(2).mean()`)
- Only new computation: `E[X]` = one `x.mean()` call
- `σ² = E[X²] - E[X]²` = one subtraction

**Information captured by each moment:**
| Moment | What it measures | Blind to | Example |
|--------|-----------------|----------|---------|
| L1 | Signal strength | Shape, texture | "Bright channel" |
| γ | Spatial peakedness | Absolute scale, texture | "Object center" |
| σ² | Spatial variation | Scale, peakedness pattern | "Rich texture" |

### 2.3 AdaptiveFreqRefine — Adaptive Tri-Frequency Spatial Attention

```
Input (B, C, H, W)
│
├── LF Map: AdaptiveAvgPool(H/4) → Conv → upsample
│     → WHERE are broad semantic regions
│
├── MF Map: BandPass → |·| → mean(C) → Conv 3×3  [NEW]
│     → WHERE are texture/pattern regions
│
├── HF Map: HighPass → |·| → mean(C) → Conv 3×3
│     → WHERE are edge/boundary regions
│
├── Gate = Sigmoid(α_lf · LF + α_mf · MF + α_hf · HF)  [Learnable α]
│
└── Output = Gate × X + X  (residual)
```

**Adaptive Balance — why it matters:**
- P3 (small objects, 80×80): edges define small objects → α_hf learns to be large
- P4 (medium objects, 40×40): balanced → all α values similar
- P5 (large objects, 20×20): context defines large objects → α_lf learns to be large
- V1's fixed 1:1 ratio cannot capture this scale-dependent emphasis

---

## 3. ARCHITECTURE SPECIFICATION

### 3.1 Full Architecture (text diagram)

```
BACKBONE:
  Layer 0:  Conv 3×3/2              → (B, 64, 320, 320)    P1/2
  Layer 1:  Conv 3×3/2              → (B, 128, 160, 160)   P2/4
  Layer 2:  C3k2_PrismV2 ×2        → (B, 128, 160, 160)   P2/4  [TFDC(lo=5,hi=3) + FCG]
  Layer 3:  Conv 3×3/2              → (B, 256, 80, 80)     P3/8
  Layer 4:  C3k2_PrismV2 ×2        → (B, 256, 80, 80)     P3/8  [TFDC + FCG]
  Layer 5:  Conv 3×3/2              → (B, 512, 40, 40)     P4/16
  Layer 6:  C3k2_PrismV2 ×2        → (B, 512, 40, 40)     P4/16 [TFDC + FCG]
  Layer 7:  Conv 3×3/2              → (B, 1024, 20, 20)    P5/32
  Layer 8:  C3k2_PrismV2 ×2        → (B, 1024, 20, 20)    P5/32 [c3k=True: TFDC(lo=7,hi=5)]
  Layer 9:  SPPF k=5               → (B, 1024, 20, 20)    P5/32

NECK (PANet with Prism V2):
  Layer 10: Upsample 2×             → (B, 1024, 40, 40)
  Layer 11: Concat[10, 6]           → (B, 1536, 40, 40)
  Layer 12: PrismV2CSP ×2           → (B, 512, 40, 40)     P4  [TFDC + FCG + AFR]
  Layer 13: Upsample 2×             → (B, 512, 80, 80)
  Layer 14: Concat[13, 4]           → (B, 768, 80, 80)
  Layer 15: PrismV2CSP ×2           → (B, 256, 80, 80)     P3  [TFDC + FCG + AFR]
  Layer 16: Conv 3×3/2              → (B, 256, 40, 40)
  Layer 17: Concat[16, 12]          → (B, 768, 40, 40)
  Layer 18: PrismV2CSP ×2           → (B, 512, 40, 40)     P4  [TFDC + FCG + AFR]
  Layer 19: Conv 3×3/2              → (B, 512, 20, 20)
  Layer 20: Concat[19, 9]           → (B, 1536, 20, 20)
  Layer 21: PrismV2CSP ×2           → (B, 1024, 20, 20)    P5  [TFDC + FCG + AFR, c3k=True]
  Layer 22: Detect[15,18,21]        → P3/P4/P5 predictions

HEAD:
  Detect: anchor-free, decoupled cls/reg heads on P3, P4, P5
```

### 3.2 Estimated Parameters/FLOPs

| Scale | Params (est.) | FLOPs (est.) | vs V1 Params | vs V1 FLOPs | vs YOLOv11 Params |
|-------|--------------|-------------|-------------|-------------|-------------------|
| PrismV2-n | ~0.95M | ~2.2G | -14% | -12% | -64% vs 2.6M |
| PrismV2-s | ~2.6M  | ~6.5G | -13% | -13% | -72% vs 9.4M |
| PrismV2-m | ~6.8M  | ~17G  | -15% | -15% | -66% vs 20M |
| PrismV2-l | ~12M   | ~30G  | -14% | -14% | -52% vs 25M |
| PrismV2-x | ~20M   | ~47G  | -13% | -15% | -65% vs 57M |

---

## 4. DETAILED COMPARISON

### 4.1 Prism V2 vs V1

| Feature | Prism V1 | Prism V2 | Why V2 is Better |
|---------|----------|----------|-------------------|
| Frequency bands | 2 (LF+HF) | 3 (LF+MF+HF) | Dedicated texture processing |
| Band-pass filter | None | AvgPool_fine - AvgPool_coarse | Isolates mid-freq textures |
| Channel descriptors | L1 + γ | L1 + γ + σ² | Captures spatial variation |
| σ² cost | N/A | Near-zero (reuses E[X²]) | Free third descriptor |
| Spatial maps | 2 (LF+HF) | 3 (LF+MF+HF) | Texture spatial awareness |
| Freq balance | Fixed 1:1 | Learned (α_lf, α_mf, α_hf) | Scale-adaptive emphasis |
| DWConv FLOPs | 17c (k=5,3) | 14.3c (k=5,3) | 16% fewer FLOPs |
| Channel split | c/2 | c/3 | More efficient per band |

### 4.2 Prism V2 vs All Architectures

| Feature | Phoenix | Chimera | Nexus | Prism V1 | **Prism V2** |
|---------|---------|---------|-------|----------|-------------|
| Spatial decomp | Scale (k=3,5) | Dilation (d=1,2,4) | Direction (I/H/V/D) | Frequency 2-band | **Frequency 3-band** |
| Channel attn | DualPool | SpectralGate | NormRatio | MCG (L1+γ) | **FCG (L1+γ+σ²)** |
| Texture aware | No | No | No | Partial | **Yes (MF band)** |
| Neck spatial | SRM | CSM | PCR | FASR (2-map) | **AFR (3-map+α)** |
| Freq awareness | None | None | None | 2-band | **3-band** |
| Adaptive balance | No | No | No | No | **Yes (learnable α)** |
| DWConv FLOPs/pixel | 17c | 9c | 7c | 17c | **14.3c** |

---

## 5. IMPLEMENTATION

### 5.1 Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `ultralytics/nn/modules/prism_v2_blocks.py` | Created | Core V2 modules |
| `ultralytics/cfg/models/11/yolo11-Prism/yolo11-PrismV2.yaml` | Created | Model config |
| `ultralytics/nn/modules/__init__.py` | Modified | Import + __all__ |
| `ultralytics/nn/tasks.py` | Modified | Module registration |
| `test_yolo_prism_v2.py` | Created | Test script |
| `plans/yolo-prism-v2-architecture.md` | Created | This document |

### 5.2 Module API

```python
from ultralytics.nn.modules.prism_v2_blocks import (
    TriFreqConv,         # Tri-band frequency decomposed DWConv
    FreqContrastGate,    # Tri-moment channel attention (L1+γ+σ²)
    PrismV2Bottleneck,   # Expand → TFDC → FCG → Project → Residual
    C3k2_PrismV2,        # C2f backbone container
    PrismV2CSP,          # C2f neck container + AFR
    AdaptiveFreqRefine,  # Tri-frequency spatial attention
)
```

### 5.3 YAML Configuration

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2_PrismV2, [128, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2_PrismV2, [256, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2_PrismV2, [512, False]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2_PrismV2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  # FPN + PAN with PrismV2CSP
  - ... PrismV2CSP with adaptive tri-frequency spatial attention
  - [[15, 18, 21], 1, Detect, [nc]]
```

---

## 6. TRAINING RECOMMENDATIONS

### 6.1 BCCD Dataset
- Use PrismV2-s or PrismV2-n scale
- The MF band should help with blood cell texture discrimination
- Expect better cell vs background separation from tri-band features

### 6.2 Hyperparameters
- Standard YOLO training: 100 epochs, batch 16, imgsz 640
- lr0=0.01, lrf=0.01, warmup_epochs=3
- No special tuning needed — V2 is drop-in compatible

### 6.3 Expected Improvements over V1
- **Fewer parameters**: ~14% reduction from c/3 split
- **Better classification**: dedicated MF band for textures
- **Scale-adaptive**: learnable α parameters adapt per pyramid level
- **Better small object detection**: α_hf learns to emphasize edges at P3
- **Richer channel info**: σ² captures spatial structure invisible to V1

---

## 7. ABLATION STUDY PLAN

| Experiment | What to Test | Expected Result |
|------------|-------------|-----------------|
| V2 → V1 DWConv (TFDC → DFDC) | Value of MF band | Lower mAP, proves MF importance |
| Remove σ² from FCG | Value of freq variance | Slight mAP drop, proves σ² helps |
| Fix α=1 in AFR | Value of learned balance | mAP drops, proves adaptation helps |
| Remove AFR entirely | Value of spatial attention | Significant mAP drop |
| Band-pass kernel sweep (3/5, 5/9, 7/11) | Optimal frequency cutoffs | Find sweet spot |
| V2 vs V1 on BCCD | Direct comparison | V2 should match/beat with fewer params |
| V2 vs baseline YOLO11 | Value of frequency design | Fewer params, competitive mAP |

---

## 8. MATHEMATICAL FOUNDATIONS

### 8.1 Band-Pass Filter from AvgPool Pair

Given two AvgPool filters with kernel sizes k_fine < k_coarse:

```
LP_fine(x)   = (1/k_f²) Σ_{i,j∈[-k_f/2, k_f/2]} x(m+i, n+j)
LP_coarse(x) = (1/k_c²) Σ_{i,j∈[-k_c/2, k_c/2]} x(m+i, n+j)

BP(x) = LP_fine(x) - LP_coarse(x)
```

In frequency domain:
- `H_fine(ω) ≈ sinc(k_f · ω / 2)` — passes frequencies up to ~π/k_f
- `H_coarse(ω) ≈ sinc(k_c · ω / 2)` — passes frequencies up to ~π/k_c
- `H_BP(ω) = H_fine(ω) - H_coarse(ω)` — passes frequencies in [π/k_c, π/k_f]

For k_fine=3, k_coarse=7: band-pass captures frequencies ω ∈ [π/7, π/3]
This corresponds to spatial patterns with periods 6-14 pixels — exactly textures!

### 8.2 Tri-Moment Channel Descriptor

The three moments capture orthogonal information:

```
L1 = E[|X|]           ← First absolute moment (scale)
γ  = √E[X²]/E[|X|]   ← Cauchy-Schwarz ratio (shape)
σ² = E[X²] - E[X]²   ← Central second moment (variation)
```

These three statistics are sufficient to characterize:
1. The magnitude of the channel (L1)
2. Whether activation is concentrated or spread (γ)
3. Whether the channel has spatial structure or is flat (σ²)

No existing attention mechanism uses all three simultaneously.

### 8.3 Frequency-Adaptive Balance

The learnable α parameters implement a soft frequency selector:

```
gate(x,y) = σ(α_lf · M_lf + α_mf · M_mf + α_hf · M_hf)
```

During backpropagation, the gradient w.r.t. α_k is:
```
∂L/∂α_k = ∂L/∂gate · σ'(·) · M_k
```

This means α_k increases when M_k is aligned with the loss gradient —
i.e., when the k-th frequency band provides useful spatial information.
Different pyramid levels naturally learn different α distributions.
