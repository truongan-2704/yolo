# YOLO-EDGE: Efficient Lightweight Detection with Gated Evolution

## Architecture Overview

**YOLO-EDGE** is a novel hybrid YOLO architecture designed for **lightweight, efficient, and accurate** object detection. It combines proven techniques from multiple state-of-the-art papers while maintaining full compatibility with the Ultralytics YOLO ecosystem.

```
┌─────────────────────────────────────────────────────────────┐
│                      YOLO-EDGE Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Backbone: C3k2_Faster (PConv from FasterNet CVPR'23)       │
│     → Processes only 1/4 channels spatially → 5.8× fewer    │
│       FLOPs per bottleneck vs standard Conv                  │
│                                                              │
│  Neck: VoVGSCSP (GSConv from SlimNeck 2022)                │
│     → Conv(1×1) + DWConv(3×3) + Shuffle → 50% fewer FLOPs  │
│       with better cross-channel mixing than pure DWConv      │
│                                                              │
│  Head: Standard Detect (P3/P4/P5 anchor-free)              │
│     → Proven detection head, no modification needed          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Design Philosophy

| Component | Design Choice | Why |
|-----------|--------------|-----|
| **Backbone** | PConv (Partial Conv) | Backbone processes high-resolution features → needs SPEED. PConv applies 3×3 conv on only 25% of channels → ~5.8× fewer FLOPs |
| **Neck** | GSConv (Group-Shuffle Conv) | Neck fuses multi-scale features → needs ACCURACY. GSConv balances efficiency (50% fewer FLOPs) with representation (1×1 + DWConv + Shuffle provides cross-channel mixing) |
| **Head** | Standard Detect | Anchor-free YOLO head is already optimized. No benefit from changing it |

## Module Details

### 1. PConv — Partial Convolution (FasterNet, CVPR 2023)

```
Input (B, C, H, W)
    │
    ├── Split ──→ x1 (B, C/4, H, W)  ──→ Conv3×3 ──→ x1'
    │
    └── Split ──→ x2 (B, 3C/4, H, W)  ──→ Identity ──→ x2
    │
    Concat(x1', x2) → BatchNorm → Output (B, C, H, W)
```

**Key Insight**: Most channels carry redundant spatial information. Processing only 1/4 of channels with 3×3 convolution achieves similar representational power at **~4× fewer FLOPs** per convolution.

**FLOPs Analysis** (c channels, H×W spatial):
- Standard Conv3×3: `9 × c² × H × W`
- PConv (1/4 channels): `9 × (c/4)² × H × W` = **1/16** of standard

### 2. FasterBottleneck — PConv + Pointwise Conv

```
Input x ──→ PConv(3×3, c/4 channels) ──→ Conv(1×1, BN, SiLU) ──→ + x ──→ Output
```

**Combined FLOPs**: ~1.56c²HW vs 9c²HW for standard 3×3 bottleneck = **~5.8× fewer FLOPs**

### 3. C3k2_Faster — C2f with FasterBottleneck

```
Input ──→ Conv1×1(split) ──→ ┌─ chunk_1
                              ├─ chunk_2 ──→ FasterBN ──→ out_1
                              └─                    ──→ FasterBN ──→ out_2
                              Concat(chunk_1, chunk_2, out_1, out_2) ──→ Conv1×1 ──→ Output
```

Drop-in replacement for `C3k2` in backbone with `c3k` support:
- `c3k=False`: 3×3 PConv kernel (lightweight)
- `c3k=True`: 5×5 PConv kernel (larger receptive field, for M/L/X scales)

### 4. GSConv — Group-Shuffle Convolution (SlimNeck)

```
Input ──→ Conv(1×1) ──→ x1 (c/2 channels)
              │
              └──→ DWConv(3×3) ──→ x2 (c/2 channels)
              │
              Concat(x1, x2) ──→ Channel Shuffle(groups=2) ──→ Output
```

**Benefits over DWConv alone**:
- 1×1 Conv provides cross-channel mixing before spatial processing
- Channel shuffle after concat interleaves the two sources
- ~50% fewer FLOPs vs standard Conv, ~2× richer than pure DWConv

### 5. VoVGSCSP — C2f with GSConv Bottleneck

```
Input ──→ Conv1×1(split) ──→ ┌─ chunk_1
                              ├─ chunk_2 ──→ GSBottleneck ──→ out_1
                              └─                          ──→ GSBottleneck ──→ out_2
                              Concat(chunk_1, chunk_2, out_1, out_2) ──→ Conv1×1 ──→ Output
```

Combines three proven techniques:
1. **C2f split-concat**: multi-resolution gradient flow (YOLOv8)
2. **GSConv**: efficient Conv+DWConv+Shuffle processing (SlimNeck)
3. **VoV aggregation**: all branch outputs concatenated (VoVNet)

## Network Architecture (YAML)

```yaml
# Backbone (C3k2_Faster with PConv)
Stage 0: Conv 3×3 s=2     → 64ch  (P1/2)   # Stem
Stage 1: Conv 3×3 s=2     → 128ch (P2/4)   # Downsample
         C3k2_Faster ×2   → 128ch          # PConv feature extraction
Stage 2: Conv 3×3 s=2     → 256ch (P3/8)   # Downsample
         C3k2_Faster ×2   → 256ch          # PConv feature extraction
Stage 3: Conv 3×3 s=2     → 512ch (P4/16)  # Downsample
         C3k2_Faster ×2   → 512ch          # PConv feature extraction
Stage 4: Conv 3×3 s=2     → 1024ch (P5/32) # Downsample
         C3k2_Faster ×2   → 1024ch (c3k=True for larger RF)
         SPPF k=5         → 1024ch         # Global context

# Neck (VoVGSCSP with GSConv)
FPN Top-Down:
  Upsample + Concat(P4)   → VoVGSCSP ×2   → 512ch  (P4 features)
  Upsample + Concat(P3)   → VoVGSCSP ×3   → 256ch  (P3 features, more repeats for small objects)

PAN Bottom-Up:
  Conv s=2 + Concat(P4)   → VoVGSCSP ×2   → 512ch  (P4 features)
  Conv s=2 + Concat(P5)   → VoVGSCSP ×2   → 1024ch (P5 features)

# Head
Detect(P3, P4, P5) — standard anchor-free YOLO detection
```

## Scaling System

| Scale | depth_mul | width_mul | max_ch | Est. Params | Use Case |
|-------|-----------|-----------|--------|-------------|----------|
| **n** | 0.50 | 0.25 | 1024 | ~1M | Mobile/edge deployment |
| **s** | 0.50 | 0.50 | 1024 | ~3M | Fast inference |
| **m** | 0.50 | 1.00 | 512 | ~8M | Balanced speed-accuracy |
| **l** | 1.00 | 1.00 | 512 | ~15M | High accuracy |
| **x** | 1.00 | 1.50 | 512 | ~25M | Maximum accuracy |

## Expected Performance vs YOLO11

| Metric | YOLO11 | YOLO-EDGE | Improvement |
|--------|--------|-----------|-------------|
| Parameters | Baseline | ~30-50% fewer | ✓ Lighter |
| FLOPs (backbone) | Baseline | ~50-70% fewer | ✓ Much faster |
| FLOPs (neck) | Baseline | ~40-50% fewer | ✓ Faster |
| Inference Speed | Baseline | ~20-40% faster | ✓ Real-time+ |
| mAP@50 | Baseline | Comparable | ✓ Maintained |
| ONNX Compatible | ✓ | ✓ | No exotic ops |
| TensorRT Compatible | ✓ | ✓ | No exotic ops |

## File Structure

```
ultralytics/
├── nn/
│   └── modules/
│       └── edge_blocks.py          # Core modules: PConv, FasterBottleneck, C3k2_Faster, GSConv, VoVGSCSP
├── cfg/
│   └── models/
│       └── 11/
│           └── yolo11-EDGE/
│               └── yolo11-EDGE.yaml  # Model configuration
plans/
└── yolo-edge-architecture.md         # This document
test_yolo_edge.py                     # Validation test suite
```

## How to Use

### Training
```python
from ultralytics import YOLO

# Load YOLO-EDGE architecture
model = YOLO("ultralytics/cfg/models/11/yolo11-EDGE/yolo11-EDGE.yaml", task='detect')

# Train on your dataset
model.train(data="your_dataset.yaml", epochs=300, imgsz=640, batch=16)
```

### Inference
```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run inference
results = model.predict("path/to/images", conf=0.25)
```

## References

1. **FasterNet** (CVPR 2023): Chen et al., "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks"
   - Source of PConv (Partial Convolution)
   - Key insight: process only 1/4 of channels spatially

2. **SlimNeck** (2022): Li et al., "Slim-neck by GSConv: A better design paradigm of detector architectures"
   - Source of GSConv (Group-Shuffle Convolution)
   - Key insight: Conv1×1 + DWConv3×3 + Shuffle for efficient fusion

3. **YOLOv8/v11** (Ultralytics): C2f split-concat architecture
   - Foundation for C3k2_Faster and VoVGSCSP wrapper modules

4. **VoVNet** (2019): Lee & Park, "An Energy and GPU-Computation Efficient Backbone Network"
   - Inspiration for VoV-style aggregation in VoVGSCSP
