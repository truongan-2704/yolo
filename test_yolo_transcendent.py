"""Smoke test cho YOLO-Transcendent.

Chạy từ repo root:
    python test_yolo_transcendent.py
"""
import torch
from ultralytics import YOLO


def main():
    yaml_path = "ultralytics/cfg/models/11/yolo11-Transcendent/yolo11-Transcendent.yaml"
    print("=" * 68)
    print("Build YOLO-Transcendent từ YAML")
    print("=" * 68)
    model = YOLO(yaml_path)
    model.info(detailed=False, verbose=True)

    print("=" * 68)
    print("Forward pass (1, 3, 640, 640)")
    print("=" * 68)
    model.model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model.model(x)
    if isinstance(out, (tuple, list)):
        feats = out[1] if isinstance(out[-1], (list, tuple)) else out
    else:
        feats = [out]
    feats = feats if isinstance(feats, (list, tuple)) else [feats]
    for i, f in enumerate(feats):
        if torch.is_tensor(f):
            print(f"  out[{i}]: {tuple(f.shape)}")
        else:
            print(f"  out[{i}]: {type(f).__name__}")
    print("PASS")


if __name__ == "__main__":
    main()
