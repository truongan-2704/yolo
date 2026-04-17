"""Test YOLO-Spectra architecture — build model and run dummy forward pass."""

from ultralytics import YOLO
import torch

def test_spectra():
    # Build model from YAML
    model = YOLO("ultralytics/cfg/models/11/yolo11-Spectra/yolo11-Spectra.yaml")
    
    # Print model info
    model.info()
    
    # Dummy forward pass
    results = model.predict(torch.randn(1, 3, 640, 640), verbose=False)
    print(f"\n✅ YOLO-Spectra forward pass successful!")
    print(f"   Predictions shape: {[r.boxes.shape for r in results]}")

if __name__ == "__main__":
    test_spectra()
