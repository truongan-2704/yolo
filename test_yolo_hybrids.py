"""Test all 5 YOLO hybrid cross-architecture variants."""

from ultralytics import YOLO
import torch

HYBRIDS = {
    "NexusPrism": "ultralytics/cfg/models/11/yolo11-Hybrid/yolo11-NexusPrism.yaml",
    "PrismEdge": "ultralytics/cfg/models/11/yolo11-Hybrid/yolo11-PrismEdge.yaml",
    "PhoenixNexus": "ultralytics/cfg/models/11/yolo11-Hybrid/yolo11-PhoenixNexus.yaml",
    "ChimeraPrism": "ultralytics/cfg/models/11/yolo11-Hybrid/yolo11-ChimeraPrism.yaml",
    "SpectraEdge": "ultralytics/cfg/models/11/yolo11-Hybrid/yolo11-SpectraEdge.yaml",
}

def test_hybrid(name, yaml_path):
    print(f"\n{'='*60}")
    print(f"Testing YOLO-{name}")
    print(f"{'='*60}")
    model = YOLO(yaml_path)
    model.info()
    results = model.predict(torch.randn(1, 3, 640, 640), verbose=False)
    print(f"✅ YOLO-{name} forward pass successful!")
    print(f"   Predictions shape: {[r.boxes.shape for r in results]}")

if __name__ == "__main__":
    for name, path in HYBRIDS.items():
        test_hybrid(name, path)
    print(f"\n{'='*60}")
    print(f"✅ All 5 hybrid variants passed!")
