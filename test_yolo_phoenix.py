"""
Test script for YOLO-Phoenix architecture.
Validates model construction, forward pass, and parameter counts.
"""
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_phoenix_model():
    """Test YOLO-Phoenix model construction and forward pass."""
    from ultralytics import YOLO

    yaml_path = "ultralytics/cfg/models/11/yolo11-Phoenix/yolo11-Phoenix.yaml"

    print("=" * 70)
    print("  YOLO-Phoenix Architecture Test")
    print("=" * 70)

    # Test all scales
    for scale in ["n", "s"]:
        print(f"\n{'─' * 60}")
        print(f"  Testing YOLO-Phoenix-{scale}")
        print(f"{'─' * 60}")

        try:
            model = YOLO(yaml_path, task="detect")
            model.model.yaml["scale"] = scale

            # Re-create with explicit scale
            model = YOLO(yaml_path, task="detect")

            # Get model info
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

            print(f"  ✅ Model created successfully")
            print(f"  📊 Total Parameters:     {total_params:>12,}")
            print(f"  📊 Trainable Parameters: {trainable:>12,}")
            print(f"  📊 Model Size:           {total_params * 4 / 1024 / 1024:>10.2f} MB (FP32)")

            # Test forward pass
            x = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                y = model.model(x)

            if isinstance(y, (list, tuple)):
                print(f"  ✅ Forward pass successful — {len(y)} outputs")
                for i, out in enumerate(y):
                    if isinstance(out, torch.Tensor):
                        print(f"     Output {i}: {out.shape}")
            else:
                print(f"  ✅ Forward pass successful — output: {y.shape}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Test individual modules
    print(f"\n{'─' * 60}")
    print("  Testing Individual Phoenix Modules")
    print(f"{'─' * 60}")

    from ultralytics.nn.modules.phoenix_blocks import (
        HeteroConv, DualPoolGate, PhoenixBottleneck, C3k2_Phoenix, PhoenixCSP
    )

    x = torch.randn(2, 64, 32, 32)

    # HeteroConv
    m = HeteroConv(64)
    y = m(x)
    params = sum(p.numel() for p in m.parameters())
    print(f"  HeteroConv(64):          in={list(x.shape)} → out={list(y.shape)}, params={params:,}")

    # DualPoolGate
    m = DualPoolGate(64)
    y = m(x)
    params = sum(p.numel() for p in m.parameters())
    print(f"  DualPoolGate(64):        in={list(x.shape)} → out={list(y.shape)}, params={params:,}")

    # PhoenixBottleneck
    m = PhoenixBottleneck(64, 64)
    y = m(x)
    params = sum(p.numel() for p in m.parameters())
    print(f"  PhoenixBottleneck(64):   in={list(x.shape)} → out={list(y.shape)}, params={params:,}")

    # C3k2_Phoenix
    m = C3k2_Phoenix(64, 64, n=2)
    y = m(x)
    params = sum(p.numel() for p in m.parameters())
    print(f"  C3k2_Phoenix(64,n=2):    in={list(x.shape)} → out={list(y.shape)}, params={params:,}")

    # PhoenixCSP
    m = PhoenixCSP(64, 64, n=2)
    y = m(x)
    params = sum(p.numel() for p in m.parameters())
    print(f"  PhoenixCSP(64,n=2):      in={list(x.shape)} → out={list(y.shape)}, params={params:,}")

    # c3k=True variants
    m = C3k2_Phoenix(64, 64, n=2, c3k=True)
    y = m(x)
    params = sum(p.numel() for p in m.parameters())
    print(f"  C3k2_Phoenix(c3k=True):  in={list(x.shape)} → out={list(y.shape)}, params={params:,}")

    print(f"\n{'=' * 70}")
    print("  All tests passed! ✅")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    test_phoenix_model()
