# YOLOv13 Architecture Design

## Overview
YOLOv13 is a next-generation object detection architecture that improves upon YOLOv11/v12 with novel modules designed for comprehensive and high-efficiency detection.

## Key Innovations

### 1. Multi-Scale Adaptive Convolution (MSConv)
- Combines depthwise convolutions at kernel sizes 3, 5, 7
- Learned per-channel adaptive mixing weights via softmax
- Captures multi-scale spatial patterns with minimal parameter overhead

### 2. Gated Channel Fusion (GCF) Attention
- Dual-path squeeze using both average and max pooling
- Gated excitation mechanism (more expressive than SE/ECA)
- Applied in bottleneck blocks for channel-wise feature refinement

### 3. Dynamic Spatial Fusion (DSF)
- Spatial attention via avg+max pooling with 7x7 conv
- Local context enhancement with depthwise + pointwise convs
- Learnable weighted blend between spatial attention and local context

### 4. YOLO13 Bottleneck (Dual Attention)
- Standard bottleneck enhanced with GCF + DSF
- Learnable residual scale (gamma parameter) for stable training
- Light variant without attention for head usage

### 5. Adaptive Downsampling (AdaptiveDown)
- Dual-path: MaxPool+1x1 and Stride-2 Conv
- Better information preservation vs simple stride-2 conv
- Replaces standard Conv downsampling in backbone

### 6. Enhanced SPPF
- SPPF with GatedChannelFusion attention post-aggregation
- Better multi-scale feature refinement

## Architecture

### Backbone
```
Conv 3x2 → Conv 3x2 → C3k2_YOLO13 → AdaptiveDown → C3k2_YOLO13 → AdaptiveDown → C3k2_YOLO13 → AdaptiveDown → C3k2_YOLO13 → EnhancedSPPF
```

### Neck (Scale-Aware FPN + PAN)
- Top-down path with YOLO13CSP blocks
- Bottom-up path with YOLO13CSP blocks  
- Cross-path channel exchange via GatedChannelFusion

### Head
- Standard Detect head (P3, P4, P5)

## Scales
| Scale | Depth | Width | Max Channels |
|-------|-------|-------|--------------|
| n     | 0.50  | 0.25  | 1024         |
| s     | 0.50  | 0.50  | 1024         |
| m     | 0.50  | 1.00  | 512          |
| l     | 1.00  | 1.00  | 512          |
| x     | 1.00  | 1.50  | 512          |

## Files
- `ultralytics/nn/modules/yolo13_blocks.py` - All novel modules
- `ultralytics/cfg/models/v13/yolov13.yaml` - Model config
- `test_yolo13.py` - Test script
