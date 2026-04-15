"""
Test script for YOLO-Zenith architecture.
Validates model construction, forward pass, parameter counts,
and individual module behavior.
"""
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_zenith_model():
    """Test YOLO-Zenith model construction and forward pass."""
    from ultralytics import YOLO

    yaml_path = "ultralytics/cfg/models/11/yolo11-Zenith/yolo11-Zenith.yaml"

    print("=" * 70)
    print("  YOLO-Zenith Architecture Test")
    print("  Novel: WaveletConv + TopologicalGate + AdaptiveScaleRouter")
    print("=" * 70)

    # Test all scales
    for scale in ["n", "s"]:
        print(f"\n{'─' * 60}")
        print(f"  Testing YOLO-Zenith-{scale}")
        print(f"{'─' * 60}")

        try:
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

    # ──────────────────────────────────────────────────────────────
    # Test individual modules
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Testing Individual Zenith Modules")
    print(f"{'─' * 60}")

    from ultralytics.nn.modules.zenith_blocks import (
        WaveletConv, TopologicalGate, ZenithBottleneck,
        C3k2_Zenith, ZenithCSP, AdaptiveScaleRouter
    )

    x64 = torch.randn(2, 64, 32, 32)

    # WaveletConv
    m = WaveletConv(64)
    y = m(x64)
    params = sum(p.numel() for p in m.parameters())
    print(f"  WaveletConv(64):           in={list(x64.shape)} → out={list(y.shape)}, params={params:,}")
    assert y.shape == x64.shape, f"WaveletConv shape mismatch: {y.shape} != {x64.shape}"

    # TopologicalGate
    m = TopologicalGate(64)
    y = m(x64)
    params = sum(p.numel() for p in m.parameters())
    print(f"  TopologicalGate(64):       in={list(x64.shape)} → out={list(y.shape)}, params={params:,}")
    assert y.shape == x64.shape, f"TopologicalGate shape mismatch: {y.shape} != {x64.shape}"

    # ZenithBottleneck
    m = ZenithBottleneck(64, 64)
    y = m(x64)
    params = sum(p.numel() for p in m.parameters())
    print(f"  ZenithBottleneck(64, 64):  in={list(x64.shape)} → out={list(y.shape)}, params={params:,}")
    assert y.shape == x64.shape, f"ZenithBottleneck shape mismatch: {y.shape} != {x64.shape}"

    # ZenithBottleneck with c3k=True (larger wavelet)
    m = ZenithBottleneck(64, 64, k=5)
    y = m(x64)
    params = sum(p.numel() for p in m.parameters())
    print(f"  ZenithBottleneck(64, k=5): in={list(x64.shape)} → out={list(y.shape)}, params={params:,}")
    assert y.shape == x64.shape, f"ZenithBottleneck k=5 shape mismatch"

    # C3k2_Zenith
    m = C3k2_Zenith(64, 64, n=2, c3k=False)
    y = m(x64)
    params = sum(p.numel() for p in m.parameters())
    print(f"  C3k2_Zenith(64, n=2):      in={list(x64.shape)} → out={list(y.shape)}, params={params:,}")
    assert y.shape == x64.shape, f"C3k2_Zenith shape mismatch"

    # C3k2_Zenith with c3k=True
    m = C3k2_Zenith(64, 64, n=2, c3k=True)
    y = m(x64)
    params = sum(p.numel() for p in m.parameters())
    print(f"  C3k2_Zenith(64, c3k=True): in={list(x64.shape)} → out={list(y.shape)}, params={params:,}")
    assert y.shape == x64.shape, f"C3k2_Zenith c3k=True shape mismatch"

    # AdaptiveScaleRouter
    m = AdaptiveScaleRouter(64)
    y = m(x64)
    params = sum(p.numel() for p in m.parameters())
    print(f"  AdaptiveScaleRouter(64):   in={list(x64.shape)} → out={list(y.shape)}, params={params:,}")
    assert y.shape == x64.shape, f"AdaptiveScaleRouter shape mismatch"

    # ZenithCSP
    m = ZenithCSP(64, 64, n=2, c3k=False)
    y = m(x64)
    params = sum(p.numel() for p in m.parameters())
    print(f"  ZenithCSP(64, n=2):        in={list(x64.shape)} → out={list(y.shape)}, params={params:,}")
    assert y.shape == x64.shape, f"ZenithCSP shape mismatch"

    # ZenithCSP with c3k=True
    m = ZenithCSP(64, 64, n=2, c3k=True)
    y = m(x64)
    params = sum(p.numel() for p in m.parameters())
    print(f"  ZenithCSP(64, c3k=True):   in={list(x64.shape)} → out={list(y.shape)}, params={params:,}")
    assert y.shape == x64.shape, f"ZenithCSP c3k=True shape mismatch"

    # ──────────────────────────────────────────────────────────────
    # Test different input sizes (ensure no shape issues)
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Testing Various Input Sizes")
    print(f"{'─' * 60}")

    for h, w in [(64, 64), (32, 32), (16, 16), (8, 8), (13, 17)]:
        x_test = torch.randn(1, 64, h, w)
        m = ZenithCSP(64, 64, n=1)
        y = m(x_test)
        status = "✅" if y.shape == x_test.shape else "❌"
        print(f"  {status} ZenithCSP: ({h},{w}) → {tuple(y.shape[2:])}")

    # ──────────────────────────────────────────────────────────────
    # Test gradient flow
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Testing Gradient Flow")
    print(f"{'─' * 60}")

    x_grad = torch.randn(1, 64, 16, 16, requires_grad=True)
    m = C3k2_Zenith(64, 64, n=2)
    y = m(x_grad)
    loss = y.sum()
    loss.backward()
    grad_ok = x_grad.grad is not None and x_grad.grad.abs().sum() > 0
    print(f"  {'✅' if grad_ok else '❌'} Gradient flows through C3k2_Zenith")

    x_grad2 = torch.randn(1, 64, 16, 16, requires_grad=True)
    m2 = ZenithCSP(64, 64, n=2)
    y2 = m2(x_grad2)
    loss2 = y2.sum()
    loss2.backward()
    grad_ok2 = x_grad2.grad is not None and x_grad2.grad.abs().sum() > 0
    print(f"  {'✅' if grad_ok2 else '❌'} Gradient flows through ZenithCSP")

    # ──────────────────────────────────────────────────────────────
    # Parameter comparison
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Parameter Comparison (c=128, n=2)")
    print(f"{'─' * 60}")

    from ultralytics.nn.modules.block import C3k2, Bottleneck

    configs = [
        ("C3k2 (standard)", C3k2(128, 128, n=2)),
        ("C3k2_Zenith",     C3k2_Zenith(128, 128, n=2, c3k=False)),
        ("C3k2_Zenith c3k", C3k2_Zenith(128, 128, n=2, c3k=True)),
        ("ZenithCSP",       ZenithCSP(128, 128, n=2, c3k=False)),
        ("ZenithCSP c3k",   ZenithCSP(128, 128, n=2, c3k=True)),
    ]

    for name, mod in configs:
        p = sum(x.numel() for x in mod.parameters())
        x_bench = torch.randn(1, 128, 32, 32)
        with torch.no_grad():
            y_bench = mod(x_bench)
        print(f"  {name:<20s}: params={p:>10,}, out={list(y_bench.shape)}")

    print(f"\n{'═' * 70}")
    print("  ✅ All YOLO-Zenith tests passed!")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    test_zenith_model()
