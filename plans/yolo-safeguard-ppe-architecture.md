# YOLO-SafeGuard: PPE-Optimized Detection Architecture

## 1. Problem Analysis

### 1.1 PPE Dataset Characteristics
- **10 classes**: helmet, vest, gloves, goggles, boots, no-helmet, no-vest, no-gloves, no-goggles, person
- **~5000 images**, baseline YOLO11 mAP ~65%
- **Deployment target**: Edge device (Jetson, mobile)

### 1.2 Key Challenges for PPE Detection

| Challenge | Why It Matters | Solution |
|---|---|---|
| **Extreme scale variation** | Goggles ~20px vs Person ~400px | P2 detection layer for tiny objects |
| **Absence detection** | no-helmet requires understanding body context | Coordinate Attention for spatial awareness |
| **Overlapping objects** | PPE items overlap on person body | BiFPN for better multi-scale fusion |
| **Edge deployment** | Must be lightweight and fast | PConv backbone ~5.8x fewer FLOPs |
| **Class imbalance** | Person class dominates, goggles rare | Focal loss + class-balanced augmentation |

### 1.3 Why Baseline Underperforms at 65% mAP

1. **No P2 layer** — YOLO11 standard only detects at P3/P4/P5. Tiny PPE like goggles/gloves at P3/8 resolution lose critical details
2. **Standard PANet** — insufficient multi-scale fusion for objects that vary 20x in size
3. **No spatial context** — absence classes (no-helmet) need to understand WHERE on the body to look

---

## 2. YOLO-SafeGuard Architecture

### 2.1 Design Principles

```
Edge-First:  PConv backbone → minimal FLOPs
PPE-Aware:   P2 layer → detect tiny goggles/gloves  
Context-Rich: CA attention → spatial body-part awareness
Multi-Scale:  BiFPN → bidirectional feature fusion
```

### 2.2 Architecture Diagram

```
Input (640×640×3)
    │
    ├── Conv 3×3/2 → 64ch ─────────────────────────── P1/2  (320×320)
    │
    ├── Conv 3×3/2 → 128ch
    ├── C3k2_Faster ×2 ────────────────────────────── P2/4  (160×160) ──→ BiFPN ──→ Detect
    │
    ├── Conv 3×3/2 → 256ch
    ├── C3k2_Faster ×2 ────────────────────────────── P3/8  (80×80)  ──→ BiFPN ──→ Detect
    │
    ├── Conv 3×3/2 → 512ch
    ├── C3k2_Faster ×2 + CA ───────────────────────── P4/16 (40×40)  ──→ BiFPN ──→ Detect
    │
    ├── Conv 3×3/2 → 1024ch
    ├── C3k2_Faster ×2 + SPPF ─────────────────────── P5/32 (20×20)  ──→ BiFPN ──→ Detect
    │
    └── Detect(P2, P3, P4, P5) — 4-scale anchor-free head
```

### 2.3 Module Choices Justified

| Module | Source | Why for PPE |
|---|---|---|
| **C3k2_Faster** (PConv) | FasterNet CVPR23, already in EDGE | 5.8x fewer FLOPs, edge-friendly |
| **BiFPN** | EfficientDet, already implemented | Bidirectional fusion critical for scale variation |
| **CoordAtt (CA)** | CVPR21, already implemented | Encodes spatial position → body-part awareness |
| **SPPF** | YOLOv5/11 standard | Multi-scale context at P5 |
| **P2 detection layer** | YOLO11-p2 variant exists | Catch tiny goggles/gloves at 160×160 resolution |

### 2.4 Novelty: SafeGuard Fusion Block

A new lightweight block combining PConv efficiency with context awareness:

```python
class SafeGuardBlock(nn.Module):
    """PConv + CA attention in a single efficient block"""
    def __init__(self, c1, c2):
        # PConv: process 1/4 channels spatially
        # CA: coordinate attention for body-part localization
        # Residual connection
```

This is novel because:
- No existing architecture combines PConv with CA in a single bottleneck
- PConv handles efficiency, CA handles context — complementary
- Specifically designed for PPE where spatial position matters

---

## 3. YAML Configuration

### 3.1 Model Config: yolo11-SafeGuard.yaml

```yaml
nc: 10  # PPE classes

backbone:
  - [-1, 1, Conv, [64, 3, 2]]              # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]             # 1-P2/4
  - [-1, 2, C3k2_Faster, [128, False]]     # 2-P2/4  PConv
  - [-1, 1, Conv, [256, 3, 2]]             # 3-P3/8
  - [-1, 2, C3k2_Faster, [256, False]]     # 4-P3/8  PConv
  - [-1, 1, Conv, [512, 3, 2]]             # 5-P4/16
  - [-1, 2, C3k2_Faster, [512, True]]      # 6-P4/16 PConv c3k=True
  - [-1, 1, Conv, [1024, 3, 2]]            # 7-P5/32
  - [-1, 2, C3k2_Faster, [1024, True]]     # 8-P5/32 PConv c3k=True
  - [-1, 1, SPPF, [1024, 5]]              # 9-P5/32
  - [-1, 1, CoordAtt, [1024]]             # 10-P5/32 Context

head: (BiFPN + P2 detection — see full YAML in implementation)
  - 4-scale detection: P2, P3, P4, P5
  - BiFPN-style bidirectional fusion
  - CA attention at fusion nodes
```

---

## 4. Training Strategy for PPE

### 4.1 Augmentation (PPE-specific)

```yaml
imgsz: 640
batch: 16
epochs: 200
optimizer: AdamW
lr0: 0.001
lrf: 0.01
warmup_epochs: 5
patience: 40

# Augmentation
mosaic: 1.0          # Multi-scale training critical for PPE
mixup: 0.1           # Mild mixup for regularization
copy_paste: 0.3      # Copy-paste rare PPE items (goggles, gloves)
degrees: 10.0        # Mild rotation — PPE orientation varies
scale: 0.5           # Scale augmentation critical for tiny objects
flipud: 0.0          # No vertical flip — people are upright
fliplr: 0.5          # Horizontal flip OK
hsv_h: 0.015         # Color jitter for different PPE colors
hsv_s: 0.7
hsv_v: 0.4
```

### 4.2 Expected Performance

| Model | Params | FLOPs | mAP50 (expected) | Edge FPS |
|---|---|---|---|---|
| YOLO11n baseline | 2.6M | 6.6G | 65% | ~45 |
| YOLO-SafeGuard-n | ~2.2M | ~5.5G | 72-75% | ~50 |
| YOLO-SafeGuard-s | ~4.5M | ~12G | 76-80% | ~35 |

Key improvements expected:
- **+5-10% mAP** from P2 layer (tiny object detection)
- **+3-5% mAP** from BiFPN (better multi-scale fusion)
- **+2-3% mAP** from CA attention (body-part context)
- **Fewer FLOPs** from PConv backbone

---

## 5. Implementation Plan

### Step 1: Create SafeGuard blocks module
- `safeguard_blocks.py`: SafeGuardBlock (PConv + CA combo)
- Reuse existing: C3k2_Faster, CoordAtt, BiFPN modules

### Step 2: Create YAML config
- `yolo11-SafeGuard/yolo11-SafeGuard.yaml` with P2 detection
- Scale variants: n/s/m

### Step 3: Register modules
- Update `__init__.py` and `tasks.py`

### Step 4: Test & validate
- Test script to verify model builds
- Parameter count and FLOPs check

### Step 5: Train on PPE dataset
- Train baseline YOLO11n on PPE (if not done)
- Train SafeGuard-n on PPE
- Compare mAP, params, FPS

---

## 6. Ablation Study Plan

| Experiment | What Changes | Expected Impact |
|---|---|---|
| Baseline YOLO11n | — | 65% mAP |
| + P2 layer | Add P2 detection | +5-8% mAP (tiny objects) |
| + BiFPN neck | Replace PANet with BiFPN | +3-5% mAP |
| + CA attention | Add CoordAtt at P4/P5 | +2-3% mAP |
| + PConv backbone | Replace C3k2 with C3k2_Faster | -30% params, similar mAP |
| Full SafeGuard | All combined | +10-15% mAP, fewer params |
