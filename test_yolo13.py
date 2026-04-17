"""Test YOLOv13 architecture - verify model loads and forward pass works."""
from ultralytics import YOLO

def test_yolo13(scale='n'):
    model_name = f'yolov13{scale}'
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    model = YOLO(f'ultralytics/cfg/models/v13/yolov13.yaml').model
    
    import torch
    x = torch.randn(1, 3, 640, 640)
    
    # Test forward pass
    with torch.no_grad():
        y = model(x)
    
    print(f"Input: {x.shape}")
    if isinstance(y, (list, tuple)):
        for i, yi in enumerate(y):
            if isinstance(yi, torch.Tensor):
                print(f"Output[{i}]: {yi.shape}")
    else:
        print(f"Output: {y.shape}")
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print("✅ SUCCESS!")

if __name__ == "__main__":
    test_yolo13('n')
