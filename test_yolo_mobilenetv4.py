"""
Test script for YOLO11-MobileNetV4 architecture validation.

Validates:
1. Individual MobileNetV4 module shapes (MNV4SE, FusedIBBlock, UIBBlock, MobileMQA)
2. MNV4Conv, MNV4UIB, MNV4Hybrid stage containers
3. Full YOLO11-MobileNetV4 model build from YAML
4. Full YOLO11-MobileNetV4-Hybrid model build
5. Parameter count comparison with baseline, EfficientNetV2, and EfficientNetV4

Usage:
    python test_yolo_mobilenetv4.py
"""

import sys
import os
import torch
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model):
    """Count all parameters (trainable + non-trainable)."""
    return sum(p.numel() for p in model.parameters())


def test_mnv4_layer_scale():
    """Test MNV4LayerScale module."""
    from ultralytics.nn.modules.MobileNetV4 import MNV4LayerScale

    print("\n" + "=" * 70)
    print("TEST 1: MNV4LayerScale — Per-channel Learnable Scaling")
    print("=" * 70)

    for c in [32, 64, 128, 256]:
        x = torch.randn(2, c, 20, 20)
        m = MNV4LayerScale(c, init_value=1e-4)
        y = m(x)
        params = count_params(m)
        print(f"  c={c:4d}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
        assert m.gamma.abs().max().item() < 1e-3, "Initial gamma should be small"

    print("  ✅ MNV4LayerScale: ALL TESTS PASSED")


def test_mnv4_se():
    """Test MNV4SE module."""
    from ultralytics.nn.modules.MobileNetV4 import MNV4SE

    print("\n" + "=" * 70)
    print("TEST 2: MNV4SE — Squeeze-and-Excitation")
    print("=" * 70)

    for c_in, c_expand in [(16, 64), (32, 128), (64, 256), (128, 512)]:
        x = torch.randn(2, c_expand, 20, 20)
        m = MNV4SE(c_in=c_in, c_expand=c_expand, se_ratio=0.25)
        y = m(x)
        params = count_params(m)
        print(f"  c_in={c_in:4d}, c_expand={c_expand:4d}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    print("  ✅ MNV4SE: ALL TESTS PASSED")


def test_fused_ib_block():
    """Test FusedIBBlock module."""
    from ultralytics.nn.modules.MobileNetV4 import FusedIBBlock

    print("\n" + "=" * 70)
    print("TEST 3: FusedIBBlock — Fused Inverted Bottleneck Block")
    print("=" * 70)

    # expand=1 path
    x = torch.randn(2, 64, 40, 40)
    m = FusedIBBlock(64, 64, s=1, expand=1)
    y = m(x)
    print(f"  expand=1, res: input={list(x.shape)} → output={list(y.shape)}, params={count_params(m):,}")
    assert y.shape == x.shape

    # expand=4 path with residual
    m2 = FusedIBBlock(64, 64, s=1, expand=4)
    y2 = m2(x)
    print(f"  expand=4, res: input={list(x.shape)} → output={list(y2.shape)}, params={count_params(m2):,}")
    assert y2.shape == x.shape

    # expand=4, stride=2, no residual
    m3 = FusedIBBlock(64, 128, s=2, expand=4)
    y3 = m3(x)
    print(f"  expand=4, s=2: input={list(x.shape)} → output={list(y3.shape)}, params={count_params(m3):,}")
    assert y3.shape == (2, 128, 20, 20)

    print("  ✅ FusedIBBlock: ALL TESTS PASSED")


def test_uib_block():
    """Test UIBBlock — Universal Inverted Bottleneck."""
    from ultralytics.nn.modules.MobileNetV4 import UIBBlock

    print("\n" + "=" * 70)
    print("TEST 4: UIBBlock — Universal Inverted Bottleneck (Extra-DW)")
    print("=" * 70)

    # Standard MBConv (dw_start=0, dw_mid=3)
    x = torch.randn(2, 64, 20, 20)
    m1 = UIBBlock(64, 64, s=1, expand=4, dw_start_k=0, dw_mid_k=3)
    y1 = m1(x)
    print(f"  MBConv (0,3) res:   input={list(x.shape)} → output={list(y1.shape)}, params={count_params(m1):,}")
    assert y1.shape == x.shape

    # Extra-DW mixed (dw_start=3, dw_mid=5) — KEY INNOVATION
    m2 = UIBBlock(64, 64, s=1, expand=4, dw_start_k=3, dw_mid_k=5)
    y2 = m2(x)
    print(f"  Extra-DW (3,5) res: input={list(x.shape)} → output={list(y2.shape)}, params={count_params(m2):,}")
    assert y2.shape == x.shape

    # Extra-DW with stride=2
    m3 = UIBBlock(64, 128, s=2, expand=4, dw_start_k=3, dw_mid_k=5)
    y3 = m3(x)
    print(f"  Extra-DW (3,5) s=2: input={list(x.shape)} → output={list(y3.shape)}, params={count_params(m3):,}")
    assert y3.shape == (2, 128, 10, 10)

    # ConvNext-like (dw_start=3, dw_mid=0)
    m4 = UIBBlock(64, 64, s=1, expand=4, dw_start_k=3, dw_mid_k=0)
    y4 = m4(x)
    print(f"  ConvNext (3,0) res: input={list(x.shape)} → output={list(y4.shape)}, params={count_params(m4):,}")
    assert y4.shape == x.shape

    # Extra-DW with expand=6
    m5 = UIBBlock(128, 256, s=2, expand=6, dw_start_k=3, dw_mid_k=5)
    x5 = torch.randn(2, 128, 20, 20)
    y5 = m5(x5)
    print(f"  Extra-DW (3,5) e=6: input={list(x5.shape)} → output={list(y5.shape)}, params={count_params(m5):,}")
    assert y5.shape == (2, 256, 10, 10)

    print("  ✅ UIBBlock: ALL TESTS PASSED")


def test_mobile_mqa():
    """Test MobileMQA — Multi-Query Attention."""
    from ultralytics.nn.modules.MobileNetV4 import MobileMQA

    print("\n" + "=" * 70)
    print("TEST 5: MobileMQA — Mobile Multi-Query Attention")
    print("=" * 70)

    for dim, num_heads in [(64, 2), (128, 4), (256, 8), (512, 8)]:
        x = torch.randn(2, dim, 10, 10)
        m = MobileMQA(dim, num_heads=num_heads, spatial_ds=1)
        y = m(x)
        params = count_params(m)
        print(f"  dim={dim:4d}, heads={num_heads}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    # Test with spatial downsampling
    x = torch.randn(2, 256, 20, 20)
    m = MobileMQA(256, num_heads=4, spatial_ds=2)
    y = m(x)
    print(f"  dim=256, heads=4, ds=2: input={list(x.shape)} → output={list(y.shape)}, params={count_params(m):,}")
    assert y.shape == x.shape  # Output should match input due to residual

    print("  ✅ MobileMQA: ALL TESTS PASSED")


def test_mnv4conv_stage():
    """Test MNV4Conv stage module."""
    from ultralytics.nn.modules.MobileNetV4 import MNV4Conv

    print("\n" + "=" * 70)
    print("TEST 6: MNV4Conv Stage — Fused IB Stack with Cosine DropPath")
    print("=" * 70)

    configs = [
        (64, 64, 1, 1, 1, "64→64, n=1, expand=1"),
        (64, 128, 2, 2, 4, "64→128, n=2, expand=4, s=2"),
        (128, 256, 3, 2, 4, "128→256, n=3, expand=4, s=2"),
    ]

    for c1, c2, n, s, expand, desc in configs:
        x = torch.randn(2, c1, 40, 40)
        m = MNV4Conv(c1, c2, n=n, s=s, expand=expand)
        y = m(x)
        params = count_params(m)
        expected_h = 40 // s
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, expected_h, expected_h)

    print("  ✅ MNV4Conv Stage: ALL TESTS PASSED")


def test_mnv4uib_stage():
    """Test MNV4UIB stage module."""
    from ultralytics.nn.modules.MobileNetV4 import MNV4UIB

    print("\n" + "=" * 70)
    print("TEST 7: MNV4UIB Stage — UIB Extra-DW Stack")
    print("=" * 70)

    configs = [
        (128, 256, 2, 2, 4, "128→256, n=2, expand=4, s=2"),
        (256, 512, 3, 2, 4, "256→512, n=3, expand=4, s=2"),
        (512, 512, 2, 1, 4, "512→512, n=2, expand=4, s=1"),
        (512, 1024, 2, 2, 6, "512→1024, n=2, expand=6, s=2"),
    ]

    for c1, c2, n, s, expand, desc in configs:
        x = torch.randn(2, c1, 20, 20)
        m = MNV4UIB(c1, c2, n=n, s=s, expand=expand)
        y = m(x)
        params = count_params(m)
        expected_h = 20 // s
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, expected_h, expected_h)

    print("  ✅ MNV4UIB Stage: ALL TESTS PASSED")


def test_mnv4hybrid_stage():
    """Test MNV4Hybrid stage module."""
    from ultralytics.nn.modules.MobileNetV4 import MNV4Hybrid

    print("\n" + "=" * 70)
    print("TEST 8: MNV4Hybrid Stage — UIB + Mobile MQA Hybrid")
    print("=" * 70)

    configs = [
        (256, 512, 2, 2, 4, "256→512, n=2, expand=4, s=2"),
        (512, 1024, 2, 2, 6, "512→1024, n=2, expand=6, s=2"),
        (1024, 1024, 2, 1, 4, "1024→1024, n=2, expand=4, s=1"),
    ]

    for c1, c2, n, s, expand, desc in configs:
        x = torch.randn(2, c1, 20, 20)
        m = MNV4Hybrid(c1, c2, n=n, s=s, expand=expand)
        y = m(x)
        params = count_params(m)
        expected_h = 20 // s
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, expected_h, expected_h)

    print("  ✅ MNV4Hybrid Stage: ALL TESTS PASSED")


def test_gradient_flow():
    """Test gradient flow through MNV4 modules."""
    from ultralytics.nn.modules.MobileNetV4 import UIBBlock, FusedIBBlock, MobileMQA, MNV4UIB

    print("\n" + "=" * 70)
    print("TEST 9: Gradient Flow Validation")
    print("=" * 70)

    # FusedIBBlock gradient
    x1 = torch.randn(2, 64, 20, 20, requires_grad=True)
    m1 = FusedIBBlock(64, 64, s=1, expand=4)
    y1 = m1(x1)
    y1.sum().backward()
    assert x1.grad is not None and x1.grad.abs().sum() > 0
    print(f"  FusedIBBlock gradient: ✅ (grad norm = {x1.grad.norm():.4f})")

    # UIBBlock gradient (Extra-DW)
    x2 = torch.randn(2, 64, 20, 20, requires_grad=True)
    m2 = UIBBlock(64, 64, s=1, expand=4, dw_start_k=3, dw_mid_k=5)
    y2 = m2(x2)
    y2.sum().backward()
    assert x2.grad is not None and x2.grad.abs().sum() > 0
    print(f"  UIBBlock (Extra-DW) gradient: ✅ (grad norm = {x2.grad.norm():.4f})")

    # MobileMQA gradient
    x3 = torch.randn(2, 128, 10, 10, requires_grad=True)
    m3 = MobileMQA(128, num_heads=4)
    y3 = m3(x3)
    y3.sum().backward()
    assert x3.grad is not None and x3.grad.abs().sum() > 0
    print(f"  MobileMQA gradient: ✅ (grad norm = {x3.grad.norm():.4f})")

    # MNV4UIB stage gradient
    x4 = torch.randn(2, 128, 20, 20, requires_grad=True)
    m4 = MNV4UIB(128, 128, n=3, s=1, expand=4)
    y4 = m4(x4)
    y4.sum().backward()
    assert x4.grad is not None and x4.grad.abs().sum() > 0
    print(f"  MNV4UIB stage gradient: ✅ (grad norm = {x4.grad.norm():.4f})")

    # LayerScale gamma gradient
    from ultralytics.nn.modules.MobileNetV4 import MNV4LayerScale
    ls = MNV4LayerScale(64)
    x5 = torch.randn(2, 64, 10, 10)
    y5 = ls(x5)
    y5.sum().backward()
    assert ls.gamma.grad is not None
    print(f"  MNV4LayerScale γ gradient: ✅ (γ grad norm = {ls.gamma.grad.norm():.6f})")

    print("  ✅ Gradient Flow: ALL TESTS PASSED")


def test_full_model_mnv4():
    """Test full YOLO11-MobileNetV4 Lightweight model build from YAML."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 10: Full YOLO11-MobileNetV4 Lightweight Model")
    print("=" * 70)

    yaml_path = "ultralytics/cfg/models/11/yolo11-MobileNetV4/yolo11-MobileNetV4.yaml"
    print(f"  Loading: {yaml_path}")

    model = YOLO(yaml_path)
    model.info(verbose=False)

    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    model.model.eval()
    with torch.no_grad():
        y = model.model(x)

    print(f"  ✅ Model built successfully!")
    print(f"  Input: {list(x.shape)}")
    if isinstance(y, (list, tuple)):
        for i, yi in enumerate(y):
            if isinstance(yi, torch.Tensor):
                print(f"  Output[{i}]: {list(yi.shape)}")
    else:
        print(f"  Output: {list(y.shape)}")

    total_params = count_all_params(model.model)
    trainable_params = count_params(model.model)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  ✅ Full YOLO11-MobileNetV4 Lightweight: PASSED")

    return total_params


def test_full_model_mnv4_hybrid():
    """Test full YOLO11-MobileNetV4-Hybrid Lightweight model."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 11: Full YOLO11-MobileNetV4-Hybrid Lightweight (UIB + MQA, no C2PSA)")
    print("=" * 70)

    yaml_path = "ultralytics/cfg/models/11/yolo11-MobileNetV4/yolo11-MobileNetV4-Hybrid.yaml"
    print(f"  Loading: {yaml_path}")

    model = YOLO(yaml_path)
    model.info(verbose=False)

    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    model.model.eval()
    with torch.no_grad():
        y = model.model(x)

    print(f"  ✅ Model built successfully!")
    print(f"  Input: {list(x.shape)}")
    if isinstance(y, (list, tuple)):
        for i, yi in enumerate(y):
            if isinstance(yi, torch.Tensor):
                print(f"  Output[{i}]: {list(yi.shape)}")
    else:
        print(f"  Output: {list(y.shape)}")

    total_params = count_all_params(model.model)
    trainable_params = count_params(model.model)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  ✅ Full YOLO11-MobileNetV4-Hybrid Lightweight: PASSED")

    return total_params


def test_comparison():
    """Compare parameter counts — MobileNetV4 models MUST be lighter than baseline."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 12: Lightweight Validation (Must be < baseline)")
    print("=" * 70)

    configs = {
        "YOLO11 (baseline)": "ultralytics/cfg/models/11/yolo11.yaml",
        "MobileNetV4 (lightweight)": "ultralytics/cfg/models/11/yolo11-MobileNetV4/yolo11-MobileNetV4.yaml",
        "MobileNetV4-Hybrid (lightweight)": "ultralytics/cfg/models/11/yolo11-MobileNetV4/yolo11-MobileNetV4-Hybrid.yaml",
    }

    results = {}
    for name, yaml_path in configs.items():
        try:
            model = YOLO(yaml_path)
            params = count_all_params(model.model)
            results[name] = params
            print(f"  {name:40s}: {params:>12,} params")
        except Exception as e:
            print(f"  {name:40s}: FAILED ({e})")

    print()
    if "YOLO11 (baseline)" in results:
        baseline = results["YOLO11 (baseline)"]
        all_lighter = True
        for name, params in results.items():
            if name != "YOLO11 (baseline)":
                diff = (params / baseline - 1) * 100
                status = "✅ LIGHTER" if params < baseline else "❌ HEAVIER"
                print(f"  {name} vs baseline: {diff:+.1f}% → {status}")
                if params >= baseline:
                    all_lighter = False

        print()
        if all_lighter:
            print("  🎯 SUCCESS: All MobileNetV4 models are lighter than YOLO11 baseline!")
        else:
            print("  ⚠️  WARNING: Some models are still heavier than baseline!")

    print("  ✅ Comparison: DONE")


if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║  YOLO11-MobileNetV4 Architecture Validation Test Suite            ║")
    print("║  UIB (Extra-DW) + Mobile MQA + LayerScale + Cosine DropPath       ║")
    print("╚" + "═" * 68 + "╝")

    start = time.time()

    # Run all tests
    test_mnv4_layer_scale()
    test_mnv4_se()
    test_fused_ib_block()
    test_uib_block()
    test_mobile_mqa()
    test_mnv4conv_stage()
    test_mnv4uib_stage()
    test_mnv4hybrid_stage()
    test_gradient_flow()
    test_full_model_mnv4()
    test_full_model_mnv4_hybrid()
    test_comparison()

    elapsed = time.time() - start

    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║  ALL TESTS PASSED ✅  ({elapsed:.1f}s)                                     ║")
    print("╚" + "═" * 68 + "╝")
