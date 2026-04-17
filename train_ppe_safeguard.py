"""
YOLO-SafeGuard Training Script for PPE Dataset
================================================
Optimized training configuration for 10-class PPE detection.

Usage:
    python train_ppe_safeguard.py

Dataset structure expected:
    data/ppe_dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
"""
from ultralytics import YOLO

if __name__ == '__main__':

    # ══════════════════════════════════════════════════════════════
    # MODEL SELECTION — Choose one:
    # ══════════════════════════════════════════════════════════════
    
    # SafeGuard (PPE-optimized, recommended)
    model = YOLO(r'ultralytics/cfg/models/11/yolo11-SafeGuard/yolo11-SafeGuard.yaml')
    
    # Baseline YOLO11 for comparison
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11.yaml')
    
    # ══════════════════════════════════════════════════════════════
    # TRAINING — Optimized for PPE dataset
    # ══════════════════════════════════════════════════════════════
    
    model.train(
        # ── Dataset ──
        data=r'data/ppe_dataset/data.yaml',    # Update path to your PPE dataset
        imgsz=640,
        
        # ── Training params ──
        batch=16,
        epochs=200,
        patience=40,                            # Early stopping patience
        
        # ── Optimizer ──
        optimizer='AdamW',                      # Better than SGD for smaller datasets
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,
        
        # ── Augmentation (PPE-optimized) ──
        mosaic=1.0,                             # Multi-scale critical for PPE
        mixup=0.1,                              # Mild mixup regularization
        copy_paste=0.3,                         # Copy-paste rare items (goggles, gloves)
        degrees=10.0,                           # Mild rotation
        scale=0.5,                              # Scale augmentation for tiny objects
        flipud=0.0,                             # No vertical flip (people are upright)
        fliplr=0.5,                             # Horizontal flip OK
        hsv_h=0.015,                            # Color jitter for PPE colors
        hsv_s=0.7,
        hsv_v=0.4,
        
        # ── Training settings ──
        amp=True,                               # Mixed precision for speed
        cache=False,
        seed=42,
        val=True,
        save_period=10,
        
        # ── Output ──
        project='runs/train_ppe',
        name='safeguard',
        workers=4,
        device=0,                               # GPU 0; change to 'cpu' if no GPU
    )
