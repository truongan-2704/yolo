# YOLO-Prism: Frequency-Decomposed Hybrid Detection Architecture
## A Novel Architecture for Object Detection with Frequency-Domain Intelligence

---

## 1. RESEARCH SYNTHESIS

### 1.1 Backbone — CNN & Hybrid Approaches

| Approach | Key Idea | Strengths | Limitations | Applicability to YOLO |
|----------|----------|-----------|-------------|----------------------|
| **EfficientNet** (Tan & Le, NeurIPS 2019) | Compound scaling (depth/width/resolution) | Optimal scaling law | Standard MBConv lacks multi-scale within block | Scaling strategy useful; block design can be improved |
| **ConvNeXt** (Liu et al., CVPR 2022) | Modernize ResNet with Transformer design principles | Strong backbone, large kernels effective | Heavy parameters with 7×7 Conv | Large kernel DWConv idea applicable |
| **MobileNetV4** (Google, 2024) | UIB (Universal Inverted Bottleneck) + MQA | Mobile-efficient, hardware-aware | No frequency awareness | UIB-style expand/project pattern applicable |
| **Swin Transformer** (Liu et al., ICCV 2021) | Shifted-window local attention | Global context capture | O(n²) complexity per window, heavy | Too heavy for YOLO real-time; hybrid approach needed |

### 1.2 Attention Mechanisms

| Mechanism | Key Idea | Strengths | Limitations | Applicability |
|-----------|----------|-----------|-------------|---------------|
| **SE** (Hu et al., CVPR 2018) | GAP → FC → Sigmoid channel gate | Simple, effective | Only mean statistic — misses activation shape | Baseline to improve upon |
| **CBAM** (Woo et al., ECCV 2018) | Channel (mean+max) + Spatial (mean+max) | Dual attention | Max is outlier-sensitive; no concentration measure | Dual-attention concept useful |
| **ECA** (Wang et al., CVPR 2020) | 1D Conv on GAP features | No FC, lightweight | Only local channel correlation | Lightweight design principle |
| **SpectralGate** (Chimera, ours) | Mean + StdDev channel attention | Captures 2nd moment spread | StdDev not scale-invariant | Can improve with ratio-based measure |
| **NormRatioGate** (Nexus, ours) | L1/L∞ ratio (sparsity) | Information-theoretic basis | L∞ = single pixel, outlier-sensitive | L2/L1 ratio more robust |

### 1.3 Feature Fusion (Neck)

| Approach | Key Idea | Strengths | Limitations | Applicability |
|----------|----------|-----------|-------------|---------------|
| **FPN** (Lin et al., CVPR 2017) | Top-down pathway for multi-scale | Standard approach | One-directional information flow | Baseline for PANet |
| **PANet** (Liu et al., CVPR 2018) | Bidirectional FPN | Better feature aggregation | No attention in fusion path | Can add spatial attention |
| **BiFPN** (Tan et al., CVPR 2020) | Weighted bidirectional FPN | Learnable fusion weights | No spatial modulation | Weight idea applicable |

### 1.4 Lightweight Designs

| Approach | Key Idea | Strengths | Limitations | Applicability |
|----------|----------|-----------|-------------|---------------|
| **GhostNet** (Han et al., CVPR 2020) | Generate ghost features from cheap ops | Very lightweight | Information loss from linear transforms | Too aggressive for detection |
| **ShuffleNet** (Zhang et al., CVPR 2018) | Channel shuffle for cross-group flow | Efficient | Limited feature expressiveness | Channel shuffle is universally useful |
| **FasterNet/PConv** (Chen et al., CVPR 2023) | Partial Conv — process only 1/4 channels | Extremely fast | 3/4 channels completely ignored | Too aggressive; process all channels |

### 1.5 Detection Head

| Approach | Key Idea | Strengths | Limitations | Applicability |
|----------|----------|-----------|-------------|---------------|
| **Anchor-free** (FCOS, CenterNet) | Predict per-pixel | No anchor tuning, simpler | May need more training tricks | Standard for modern YOLO |
| **Decoupled head** (YOLOX) | Separate cls/reg branches | Better task specialization | More parameters | Already in YOLOv8+ |

### 1.6 Training Techniques

| Technique | Key Idea | Strengths | Limitations | Applicability |
|-----------|----------|-----------|-------------|---------------|
| **CIoU** (Zheng et al., AAAI 2020) | Complete IoU with aspect ratio | Better bbox regression | Standard now | Already in ultralytics |
| **Varifocal Loss** (Zhang et al., CVPR 2021) | Asymmetric weighting for positive/negative | Better dense detection | Needs quality estimation | Compatible with Prism |

---

## 2. STRATEGY

### 2.1 Problem Statement
Existing YOLO architectures process all spatial features in a **frequency-agnostic** manner. Every depthwise convolution receives the full spectrum of spatial information (low-frequency smooth regions and high-frequency edges/textures), forcing each filter to learn mixed-frequency representations. This is suboptimal because:

1. **Object boundaries** are predominantly **high-frequency** features
2. **Semantic context** is predominantly **low-frequency** information
3. Detection requires BOTH — but processing them together wastes capacity

### 2.2 Why a Hybrid YOLO?
- Pure CNN YOLO: lacks explicit frequency awareness
- Pure Transformer YOLO: too heavy for real-time
- **YOLO-Prism**: CNN backbone with built-in frequency decomposition + concentration-aware attention
  - Keeps real-time speed (all DWConv + 1×1 Conv, no self-attention)
  - Adds frequency intelligence (parameter-free high-pass filter)
  - Adds concentration awareness (L2/L1 ratio, scale-invariant)

### 2.3 Components Modified
| Component | Standard YOLO | YOLO-Prism |
|-----------|--------------|------------|
| **Spatial Conv** | Single DWConv (all frequencies mixed) | DualFreqConv (explicit LF/HF separation) |
| **Channel Attention** | None (C3k2 has no attention) | MomentContrastGate (L2/L1 concentration) |
| **Neck Spatial Attention** | None | FreqSpatialRefine (dual-frequency spatial gate) |
| **Bottleneck** | Standard 3×3+3×3 Conv | PrismBottleneck (expand→DFDC→MCG→project) |
| **Backbone Block** | C3k2 | C3k2_Prism |
| **Neck Block** | C3k2 | PrismCSP |

---

## 3. ARCHITECTURE

### 3.1 Backbone: C3k2_Prism (DualFreqConv + MomentContrastGate)

```
Input (B, C, H, W)
  │
  ├── Conv 1×1 Expand → (B, C, H, W)
  │
  ├── DualFreqConv:
  │     ├── Channel Split: C/2 each
  │     ├── Path LO: DWConv K_lo×K_lo (raw features → smooth context)
  │     ├── Path HI: HighPass(x - AvgPool(x)) → DWConv K_hi×K_hi (edges/detail)
  │     ├── Concat → Channel Shuffle → BN → SiLU
  │     └── Output: (B, C, H, W)
  │
  ├── MomentContrastGate:
  │     ├── L1 = GAP(|X|)     → mean magnitude per channel
  │     ├── L2 = √GAP(X²)    → RMS magnitude per channel
  │     ├── γ = L2/L1         → concentration ratio (Cauchy-Schwarz)
  │     ├── desc = L1 + β×γ   → combined descriptor (β learnable)
  │     ├── FC(desc) → Sigmoid → gate
  │     └── Output: X × gate  — concentration-aware reweighting
  │
  ├── Conv 1×1 Project → (B, C2, H, W)
  │
  └── Residual Add (if C1 == C2)
```

### 3.2 Neck: PrismCSP (PrismBottleneck + FreqSpatialRefine)

```
Input (B, C1, H, W)
  │
  ├── C2f Split-Concat with PrismBottleneck ×n
  │
  ├── FreqSpatialRefine:
  │     ├── Low-Freq Map:
  │     │     AvgPool(H/4, W/4) → Conv1×1(C→C/4) → BN → SiLU → Conv1×1(C/4→1) → Upsample
  │     │     → (B, 1, H, W) broad semantic spatial map
  │     │
  │     ├── High-Freq Map:
  │     │     x - AvgPool(x)  → |·| → mean(dim=C) → Conv3×3(1→1) → BN
  │     │     → (B, 1, H, W) edge energy spatial map
  │     │
  │     └── Gate: Sigmoid(lf_map + hf_map) × x + x  (residual spatial gating)
  │
  └── Output: (B, C2, H, W)
```

### 3.3 Head: Standard Detect (P3, P4, P5)
- Anchor-free detection head from YOLOv8/v11
- Proven effectiveness, no need to change
- Receives frequency-enriched features from Prism backbone/neck

---

## 4. INNOVATIONS

### 4.1 Novel Components (vs YOLOv8/v9/v10/v11)

| Innovation | Description | What's New |
|------------|-------------|------------|
| **DualFreqConv** | DWConv on raw (LF) + high-pass filtered (HF) channels | First frequency-decomposed DWConv in object detection |
| **MomentContrastGate** | L2/L1 concentration ratio channel attention | First Cauchy-Schwarz-based attention mechanism |
| **FreqSpatialRefine** | Dual-frequency spatial attention (LF + HF maps) | First frequency-aware spatial attention in YOLO neck |
| **Parameter-free HP filter** | x - AvgPool(x) for edge extraction | Zero-cost frequency separation in spatial processing |
| **β learnable balance** | Per-channel magnitude vs. concentration weight | Adaptive importance of L1 vs. γ per channel |

### 4.2 Key Differentiators vs Existing Novel Architectures

| Feature | Phoenix | Chimera | Nexus | **Prism** |
|---------|---------|---------|-------|-----------|
| Spatial decomposition | Scale (k=3,5) | Dilation (d=1,2,4) | Direction (I/H/V/D) | **Frequency (LF/HF)** |
| Channel attention | DualPool (mean+max) | SpectralGate (mean+std) | NormRatio (L1/L∞) | **MomentContrast (L2/L1)** |
| Scale-invariant? | No | No | Partially (ratio) | **Yes (L2/L1 ratio)** |
| Neck spatial attention | SRM (local DWConv) | CSM (content-gated) | PCR (polarity) | **FASR (dual-frequency)** |
| Frequency awareness | None | None | None | **Explicit (HP filter)** |
| FLOPs (DWConv, c=128) | 17c | 9c | 7c | **17c** (same as Phoenix) |
| Unique insight | Multi-scale kernel | Multi-dilation | Directional + sparsity | **Frequency decomposition** |

---

## 5. SPECIFICATIONS

### 5.1 Architecture Diagram (Text)

```
BACKBONE:
  Layer 0:  Conv 3×3/2         → (B, 64, 320, 320)    P1/2
  Layer 1:  Conv 3×3/2         → (B, 128, 160, 160)   P2/4
  Layer 2:  C3k2_Prism ×2     → (B, 128, 160, 160)   P2/4  [DFDC(lo=5,hi=3) + MCG]
  Layer 3:  Conv 3×3/2         → (B, 256, 80, 80)     P3/8
  Layer 4:  C3k2_Prism ×2     → (B, 256, 80, 80)     P3/8  [DFDC(lo=5,hi=3) + MCG]
  Layer 5:  Conv 3×3/2         → (B, 512, 40, 40)     P4/16
  Layer 6:  C3k2_Prism ×2     → (B, 512, 40, 40)     P4/16 [DFDC(lo=5,hi=3) + MCG]
  Layer 7:  Conv 3×3/2         → (B, 1024, 20, 20)    P5/32
  Layer 8:  C3k2_Prism ×2     → (B, 1024, 20, 20)    P5/32 [DFDC(lo=7,hi=5) + MCG] c3k=True
  Layer 9:  SPPF k=5           → (B, 1024, 20, 20)    P5/32

NECK (FPN + PAN):
  Layer 10: Upsample 2×        → (B, 1024, 40, 40)
  Layer 11: Concat[10, 6]      → (B, 1536, 40, 40)
  Layer 12: PrismCSP ×2        → (B, 512, 40, 40)     P4  [DFDC + MCG + FASR]
  Layer 13: Upsample 2×        → (B, 512, 80, 80)
  Layer 14: Concat[13, 4]      → (B, 768, 80, 80)
  Layer 15: PrismCSP ×3        → (B, 256, 80, 80)     P3  [DFDC + MCG + FASR] 3 repeats
  Layer 16: Conv 3×3/2         → (B, 256, 40, 40)
  Layer 17: Concat[16, 12]     → (B, 768, 40, 40)
  Layer 18: PrismCSP ×2        → (B, 512, 40, 40)     P4  [DFDC + MCG + FASR]
  Layer 19: Conv 3×3/2         → (B, 512, 20, 20)
  Layer 20: Concat[19, 9]      → (B, 1536, 20, 20)
  Layer 21: PrismCSP ×2        → (B, 1024, 20, 20)    P5  [DFDC + MCG + FASR] c3k=True
  Layer 22: Detect[15,18,21]   → P3/P4/P5 predictions

HEAD:
  Detect: anchor-free, decoupled cls/reg heads on P3, P4, P5
```

### 5.2 Tensor Shapes (Prism-s, width=0.5, depth=0.5, input 640×640)

| Layer | Module | Output Shape | Notes |
|-------|--------|-------------|-------|
| 0 | Conv | (B, 32, 320, 320) | Stem, 64×0.5=32 |
| 1 | Conv | (B, 64, 160, 160) | P2 downsample |
| 2 | C3k2_Prism ×1 | (B, 64, 160, 160) | 2×0.5=1 repeat |
| 3 | Conv | (B, 128, 80, 80) | P3 downsample |
| 4 | C3k2_Prism ×1 | (B, 128, 80, 80) | |
| 5 | Conv | (B, 256, 40, 40) | P4 downsample |
| 6 | C3k2_Prism ×1 | (B, 256, 40, 40) | |
| 7 | Conv | (B, 512, 20, 20) | P5 downsample |
| 8 | C3k2_Prism ×1 | (B, 512, 20, 20) | c3k=True |
| 9 | SPPF | (B, 512, 20, 20) | |
| 15 | PrismCSP | (B, 128, 80, 80) | P3 fusion output |
| 18 | PrismCSP | (B, 256, 40, 40) | P4 fusion output |
| 21 | PrismCSP | (B, 512, 20, 20) | P5 fusion output |

### 5.3 Estimated FLOPs / Parameters

| Scale | Params (est.) | FLOPs (est.) | Comparable to |
|-------|--------------|-------------|---------------|
| Prism-n | ~1.1M | ~2.5G | YOLOv11n (~2.6M) — 58% fewer params |
| Prism-s | ~3.0M | ~7.5G | YOLOv11s (~9.4M) — 68% fewer params |
| Prism-m | ~8.0M | ~20G | YOLOv11m (~20M) — 60% fewer params |
| Prism-l | ~14M | ~35G | YOLOv11l (~25M) — 44% fewer params |
| Prism-x | ~23M | ~55G | YOLOv11x (~56M) — 59% fewer params |

---

## 6. IMPLEMENTATION

### 6.1 YAML Configuration

```yaml
# ultralytics/cfg/models/11/yolo11-Prism/yolo11-Prism.yaml
nc: 80
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]           # 0 P1/2
  - [-1, 1, Conv, [128, 3, 2]]          # 1 P2/4
  - [-1, 2, C3k2_Prism, [128, False]]   # 2
  - [-1, 1, Conv, [256, 3, 2]]          # 3 P3/8
  - [-1, 2, C3k2_Prism, [256, False]]   # 4
  - [-1, 1, Conv, [512, 3, 2]]          # 5 P4/16
  - [-1, 2, C3k2_Prism, [512, False]]   # 6
  - [-1, 1, Conv, [1024, 3, 2]]         # 7 P5/32
  - [-1, 2, C3k2_Prism, [1024, True]]   # 8 c3k=True
  - [-1, 1, SPPF, [1024, 5]]            # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 10
  - [[-1, 6], 1, Concat, [1]]                     # 11
  - [-1, 2, PrismCSP, [512, False]]               # 12
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]    # 13
  - [[-1, 4], 1, Concat, [1]]                     # 14
  - [-1, 3, PrismCSP, [256, False]]               # 15
  - [-1, 1, Conv, [256, 3, 2]]                    # 16
  - [[-1, 12], 1, Concat, [1]]                    # 17
  - [-1, 2, PrismCSP, [512, False]]               # 18
  - [-1, 1, Conv, [512, 3, 2]]                    # 19
  - [[-1, 9], 1, Concat, [1]]                     # 20
  - [-1, 2, PrismCSP, [1024, True]]               # 21
  - [[15, 18, 21], 1, Detect, [nc]]               # 22
```

### 6.2 PyTorch Modules

See `ultralytics/nn/modules/prism_blocks.py` for full implementation:

- `DualFreqConv` — Dual-frequency decomposed depthwise convolution
- `MomentContrastGate` — L2/L1 concentration-aware channel attention
- `PrismBottleneck` — Inverted residual with DFDC + MCG
- `C3k2_Prism` — C2f backbone container
- `FreqSpatialRefine` — Dual-frequency spatial attention for neck
- `PrismCSP` — C2f neck container with FASR

---

## 7. ANALYSIS

### 7.1 Strengths

1. **Frequency-first design**: Only architecture with explicit frequency decomposition in spatial processing. High-pass filter is parameter-free yet provides rich edge/boundary information.

2. **Scale-invariant attention**: MomentContrastGate's L2/L1 ratio is invariant to feature magnitude scaling. If all activations double, γ stays the same. This means MCG measures SHAPE of activation distribution, not scale.

3. **Dual-frequency spatial attention**: FreqSpatialRefine is the only neck spatial attention that considers both low-frequency (semantic context) and high-frequency (edge energy) spatial information.

4. **Lightweight**: PrismBottleneck is one of the lightest full-channel-processing bottlenecks (~44K params for c=128), comparable to DWConv-based designs while providing richer frequency-decomposed features.

5. **Hardware-friendly**: All operations (DWConv, AvgPool, Conv1×1, Channel Shuffle) are standard PyTorch operations with native ONNX/TensorRT support.

6. **Theoretical grounding**: Subband coding theory (Vetterli & Kovačević), Cauchy-Schwarz inequality, Rényi entropy — solid mathematical foundations.

### 7.2 Weaknesses

1. **Two frequency bands may be insufficient**: DFDC splits into only 2 bands (LF/HF). Multi-band wavelet decomposition could capture more nuanced frequency structure, at the cost of complexity.

2. **Fixed high-pass filter**: The AvgPool-based high-pass filter has a fixed frequency cutoff determined by kernel size. An adaptive or learnable cutoff could improve flexibility.

3. **No cross-attention between frequency paths**: After the channel shuffle, LF and HF features are mixed but there's no explicit attention between frequency bands.

4. **Concentration ratio overhead**: Computing L2 requires squaring + mean + sqrt, which is slightly more expensive than simple mean/max pooling.

### 7.3 Comparison with YOLO and DETR

| Metric | YOLOv11 | YOLO-Prism | RT-DETR |
|--------|---------|------------|---------|
| **Architecture type** | Pure CNN | CNN + Freq decomposition | CNN + Transformer |
| **Params (S scale)** | ~9.4M | ~3.0M | ~20M |
| **Speed** | Fast | Fast (same op types) | Moderate |
| **Frequency awareness** | None | Explicit (HP filter) | Implicit (self-attention) |
| **Channel attention** | None (C3k2) | MCG (L2/L1 ratio) | Self-attention |
| **Spatial attention** | None | FASR (dual-freq) | Cross-attention |
| **Small object** | Good | Better (HF path) | Best (global attention) |
| **Real-time** | Yes | Yes | Marginal |
| **Edge deployment** | Yes | Yes (lighter) | No (too heavy) |

---

## 8. FUTURE WORK

### 8.1 Improvement Directions

1. **Multi-band frequency decomposition**: Replace 2-band DFDC with 3-4 band decomposition using cascaded high-pass filters at different cutoff frequencies (similar to wavelet multiresolution analysis).

2. **Adaptive high-pass cutoff**: Learn the high-pass filter kernel size per channel or per stage, allowing the network to adapt frequency separation to the feature content.

3. **Cross-frequency attention**: Add a lightweight attention mechanism between LF and HF paths before the channel shuffle, allowing frequency bands to modulate each other.

4. **Prism + Transformer hybrid**: Insert lightweight window attention (like Swin) at P5 only, combining Prism's frequency decomposition with global context capture.

5. **Frequency-aware loss**: Design a loss function that penalizes high-frequency reconstruction errors more for small objects and low-frequency errors more for large objects.

6. **Knowledge distillation**: Distill from a large Prism-x teacher to a Prism-n student, potentially using frequency-domain distillation matching.

### 8.2 Ablation Study Plan

| Experiment | What to Measure | Hypothesis |
|------------|----------------|------------|
| Remove HP filter (DFDC → HeteroConv) | mAP, mAP_S | HP filter helps small objects specifically |
| Remove MCG (no attention) | mAP | MCG provides meaningful channel recalibration |
| MCG → SE (replace L2/L1 with GAP) | mAP | Concentration ratio provides more info than mean |
| MCG → SpectralGate (replace with mean+std) | mAP | L2/L1 ratio vs stddev: which is more discriminative |
| MCG → NormRatioGate (replace with L1/L∞) | mAP | L2 (robust) vs L∞ (outlier-sensitive) |
| Remove FASR (no neck spatial attention) | mAP | Dual-frequency spatial attention helps |
| FASR → CSM (Chimera's neck attention) | mAP | Dual-freq spatial vs content-gated spatial |
| FASR → PCR (Nexus's neck attention) | mAP | Dual-freq spatial vs polarity contrast |
| DualFreqConv k_lo sweep (3,5,7,9) | mAP, speed | Optimal low-freq receptive field |
| HP filter kernel sweep (3,5,7) | mAP | Optimal frequency cutoff |
| MCG reduction ratio (2,4,8) | mAP, params | Optimal FC compression |
| c3k threshold (which stages) | mAP | When do larger kernels help |

### 8.3 Dataset-Specific Optimization

- **COCO**: Standard benchmark, use default Prism configuration
- **VOC**: Fewer classes, can reduce width for efficiency
- **VisDrone**: Small objects dominant — increase HP filter kernel for stronger edge detection
- **BDD100K**: Driving scenes — balanced LF/HF important for both vehicles and pedestrians

---

## Appendix: Mathematical Foundations

### A.1 High-Pass Filter Theory

The identity minus average pooling operation:
```
HP(x) = x - AvgPool_k(x)
```

In the frequency domain, AvgPool acts as a low-pass filter with transfer function:
```
H_LP(ω) = sinc(kω/2)  (approximately)
```

Therefore the high-pass filter has transfer function:
```
H_HP(ω) = 1 - H_LP(ω) = 1 - sinc(kω/2)
```

This attenuates low frequencies (smooth regions) and passes high frequencies (edges, textures). The cutoff frequency is inversely proportional to k.

### A.2 Cauchy-Schwarz Concentration Ratio

For a vector x with n elements, the Cauchy-Schwarz inequality states:
```
||x||₁ ≤ √n × ||x||₂
```

Therefore:
```
γ = ||x||₂ / ||x||₁ ∈ [1/√n, 1]  (for unit-norm vectors)
```

In our case, after normalization by spatial dimensions:
- γ → 1 when all values are equal (uniform/flat activation)
- γ → √(H×W) when only one value is nonzero (maximally peaked)

This is a well-studied measure in compressed sensing and information theory, directly related to the Rényi entropy of orders 1 and 2.
