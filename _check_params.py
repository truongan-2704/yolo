import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO

models = {
    'baseline': 'ultralytics/cfg/models/11/yolo11.yaml',
    'MNV4': 'ultralytics/cfg/models/11/yolo11-MobileNetV4/yolo11-MobileNetV4.yaml',
    'MNV4-Hybrid': 'ultralytics/cfg/models/11/yolo11-MobileNetV4/yolo11-MobileNetV4-Hybrid.yaml',
}

for k, v in models.items():
    model = YOLO(v)
    params = sum(p.numel() for p in model.model.parameters())
    print(f'{k}: {params:,} params')
