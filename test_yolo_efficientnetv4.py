"""
Test script for YOLO11-EfficientNetV4 architecture validation.

Validates:
1. Individual V4 module shapes (CASE, MKDWConv, LayerScale)
2. FusedMBConvBlockV4 and MBConvBlockV4 forward pass
3. FusedMBConvV4 and MBConvV4 stage containers
4. Full YOLO11-EfficientNetV4 model build from YAML
5. Full EfficientNetV4 + DCNF-V6 hybrid model build
6. Parameter count comparison with baseline and EfficientNetV2

Usage:
    python test_yolo_efficientnetv4.py
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


def test_layer_scale():
    """Test LayerScale module."""
    from ultralytics.nn.modules.EfficientNetV4 import LayerScale

    print("\n" + "=" * 70)
    print("TEST 1: LayerScale — Per-channel Learnable Scaling")
    print("=" * 70)

    for c in [32, 64, 128, 256]:
        x = torch.randn(2, c, 20, 20)
        m = LayerScale(c, init_value=1e-4)
        y = m(x)
        params = count_params(m)
        print(f"  c={c:4d}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
        # Verify initial scaling is very small
        assert m.gamma.abs().max().item() < 1e-3, "Initial gamma should be small"

    print("  ✅ LayerScale: ALL TESTS PASSED")


def test_case():
    """Test Context-Aware SE (CASE) module."""
    from ultralytics.nn.modules.EfficientNetV4 import CASE

    print("\n" + "=" * 70)
    print("TEST 2: CASE — Context-Aware Squeeze-and-Excitation (Dual-pooling)")
    print("=" * 70)

    for c_in, c_expand in [(16, 64), (32, 128), (64, 256), (128, 512)]:
        x = torch.randn(2, c_expand, 20, 20)
        m = CASE(c_in=c_in, c_expand=c_expand, se_ratio=0.25)
        y = m(x)
        params = count_params(m)
        print(f"  c_in={c_in:4d}, c_expand={c_expand:4d}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    # Verify dual-pooling works (avg + max)
    m = CASE(32, 128)
    assert hasattr(m, 'avg_pool'), "CASE should have avg_pool"
    assert hasattr(m, 'max_pool'), "CASE should have max_pool"

    print("  ✅ CASE: ALL TESTS PASSED")


def test_mkdwconv():
    """Test Multi-Kernel Depthwise Conv module."""
    from ultralytics.nn.modules.EfficientNetV4 import MKDWConv

    print("\n" + "=" * 70)
    print("TEST 3: MKDWConv — Multi-Kernel Depthwise Convolution (3×3 + 5×5)")
    print("=" * 70)

    for c in [32, 64, 128, 256, 512]:
        x = torch.randn(2, c, 20, 20)
        m = MKDWConv(c, s=1)
        y = m(x)
        params = count_params(m)
        print(f"  c={c:4d} s=1: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch for stride=1: {y.shape} != {x.shape}"

    # Test with stride=2
    for c in [64, 128, 256]:
        x = torch.randn(2, c, 20, 20)
        m = MKDWConv(c, s=2)
        y = m(x)
        expected_shape = (2, c, 10, 10)
        print(f"  c={c:4d} s=2: input={list(x.shape)} → output={list(y.shape)}")
        assert y.shape == expected_shape, f"Shape mismatch for stride=2: {y.shape} != {expected_shape}"

    print("  ✅ MKDWConv: ALL TESTS PASSED")


def test_fused_mbconv_v4_block():
    """Test FusedMBConvBlockV4 module."""
    from ultralytics.nn.modules.EfficientNetV4 import FusedMBConvBlockV4

    print("\n" + "=" * 70)
    print("TEST 4: FusedMBConvBlockV4 — Enhanced Fused-MBConv Block")
    print("=" * 70)

    # expand=1 path
    x = torch.randn(2, 64, 40, 40)
    m = FusedMBConvBlockV4(64, 64, s=1, expand=1)
    y = m(x)
    print(f"  expand=1, res: input={list(x.shape)} → output={list(y.shape)}, params={count_params(m):,}")
    assert y.shape == x.shape

    # expand=4 path with residual
    m2 = FusedMBConvBlockV4(64, 64, s=1, expand=4)
    y2 = m2(x)
    print(f"  expand=4, res: input={list(x.shape)} → output={list(y2.shape)}, params={count_params(m2):,}")
    assert y2.shape == x.shape

    # expand=4, stride=2, no residual
    m3 = FusedMBConvBlockV4(64, 128, s=2, expand=4)
    y3 = m3(x)
    print(f"  expand=4, s=2: input={list(x.shape)} → output={list(y3.shape)}, params={count_params(m3):,}")
    assert y3.shape == (2, 128, 20, 20)

    # Verify LayerScale exists on residual blocks
    m_res = FusedMBConvBlockV4(64, 64, s=1, expand=4)
    assert hasattr(m_res, 'ls'), "Should have LayerScale on residual path"

    print("  ✅ FusedMBConvBlockV4: ALL TESTS PASSED")


def test_mbconv_v4_block():
    """Test MBConvBlockV4 module."""
    from ultralytics.nn.modules.EfficientNetV4 import MBConvBlockV4

    print("\n" + "=" * 70)
    print("TEST 5: MBConvBlockV4 — MBConv with CASE + MKDWConv + LayerScale")
    print("=" * 70)

    # expand=4, residual
    x = torch.randn(2, 64, 20, 20)
    m = MBConvBlockV4(64, 64, s=1, expand=4)
    y = m(x)
    print(f"  expand=4, res: input={list(x.shape)} → output={list(y.shape)}, params={count_params(m):,}")
    assert y.shape == x.shape

    # expand=4, stride=2
    m2 = MBConvBlockV4(64, 128, s=2, expand=4)
    y2 = m2(x)
    print(f"  expand=4, s=2: input={list(x.shape)} → output={list(y2.shape)}, params={count_params(m2):,}")
    assert y2.shape == (2, 128, 10, 10)

    # expand=6
    m3 = MBConvBlockV4(128, 256, s=2, expand=6)
    x3 = torch.randn(2, 128, 20, 20)
    y3 = m3(x3)
    print(f"  expand=6, s=2: input={list(x3.shape)} → output={list(y3.shape)}, params={count_params(m3):,}")
    assert y3.shape == (2, 256, 10, 10)

    print("  ✅ MBConvBlockV4: ALL TESTS PASSED")


def test_fused_mbconv_v4_stage():
    """Test FusedMBConvV4 stage module."""
    from ultralytics.nn.modules.EfficientNetV4 import FusedMBConvV4

    print("\n" + "=" * 70)
    print("TEST 6: FusedMBConvV4 Stage — Stack with Cosine DropPath")
    print("=" * 70)

    configs = [
        (64, 64, 1, 1, 1, "64→64, n=1, expand=1"),
        (64, 128, 2, 2, 4, "64→128, n=2, expand=4, s=2"),
        (128, 256, 3, 2, 4, "128→256, n=3, expand=4, s=2"),
    ]

    for c1, c2, n, s, expand, desc in configs:
        x = torch.randn(2, c1, 40, 40)
        m = FusedMBConvV4(c1, c2, n=n, s=s, expand=expand)
        y = m(x)
        params = count_params(m)
        expected_h = 40 // s
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, expected_h, expected_h)

    print("  ✅ FusedMBConvV4 Stage: ALL TESTS PASSED")


def test_mbconv_v4_stage():
    """Test MBConvV4 stage module."""
    from ultralytics.nn.modules.EfficientNetV4 import MBConvV4

    print("\n" + "=" * 70)
    print("TEST 7: MBConvV4 Stage — Stack with CASE + MKDWConv + Cosine")
    print("=" * 70)

    configs = [
        (128, 256, 2, 2, 4, "128→256, n=2, expand=4, s=2"),
        (256, 512, 3, 2, 4, "256→512, n=3, expand=4, s=2"),
        (512, 512, 2, 1, 4, "512→512, n=2, expand=4, s=1"),
        (512, 1024, 2, 2, 6, "512→1024, n=2, expand=6, s=2"),
    ]

    for c1, c2, n, s, expand, desc in configs:
        x = torch.randn(2, c1, 20, 20)
        m = MBConvV4(c1, c2, n=n, s=s, expand=expand)
        y = m(x)
        params = count_params(m)
        expected_h = 20 // s
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, expected_h, expected_h)

    print("  ✅ MBConvV4 Stage: ALL TESTS PASSED")


def test_gradient_flow():
    """Test gradient flow through V4 modules."""
    from ultralytics.nn.modules.EfficientNetV4 import MBConvBlockV4, FusedMBConvBlockV4, MBConvV4

    print("\n" + "=" * 70)
    print("TEST 8: Gradient Flow Validation")
    print("=" * 70)

    # FusedMBConvBlockV4 gradient
    x1 = torch.randn(2, 64, 20, 20, requires_grad=True)
    m1 = FusedMBConvBlockV4(64, 64, s=1, expand=4)
    y1 = m1(x1)
    y1.sum().backward()
    assert x1.grad is not None and x1.grad.abs().sum() > 0
    print(f"  FusedMBConvBlockV4 gradient: ✅ (grad norm = {x1.grad.norm():.4f})")

    # MBConvBlockV4 gradient
    x2 = torch.randn(2, 64, 20, 20, requires_grad=True)
    m2 = MBConvBlockV4(64, 64, s=1, expand=4)
    y2 = m2(x2)
    y2.sum().backward()
    assert x2.grad is not None and x2.grad.abs().sum() > 0
    print(f"  MBConvBlockV4 gradient: ✅ (grad norm = {x2.grad.norm():.4f})")

    # MBConvV4 stage gradient
    x3 = torch.randn(2, 128, 20, 20, requires_grad=True)
    m3 = MBConvV4(128, 128, n=3, s=1, expand=4)
    y3 = m3(x3)
    y3.sum().backward()
    assert x3.grad is not None and x3.grad.abs().sum() > 0
    print(f"  MBConvV4 stage gradient: ✅ (grad norm = {x3.grad.norm():.4f})")

    # LayerScale gamma gradient
    from ultralytics.nn.modules.EfficientNetV4 import LayerScale
    ls = LayerScale(64)
    x4 = torch.randn(2, 64, 10, 10)
    y4 = ls(x4)
    y4.sum().backward()
    assert ls.gamma.grad is not None
    print(f"  LayerScale γ gradient: ✅ (γ grad norm = {ls.gamma.grad.norm():.6f})")

    print("  ✅ Gradient Flow: ALL TESTS PASSED")


def test_full_model_v4():
    """Test full YOLO11-EfficientNetV4 model build from YAML."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 9: Full YOLO11-EfficientNetV4 Model (C3k2 head)")
    print("=" * 70)

    yaml_path = "ultralytics/cfg/models/11/yolo11-EfficientNetV4/yolo11-EfficientNetV4.yaml"
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
    print(f"  ✅ Full YOLO11-EfficientNetV4: PASSED")

    return total_params


def test_full_model_v4_dcnf():
    """Test full YOLO11-EfficientNetV4 + DCNF-V6 model."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 10: Full EfficientNetV4 + DCNF-V6 Hybrid Model")
    print("=" * 70)

    yaml_path = "ultralytics/cfg/models/11/yolo11-EfficientNetV4/yolo11-EfficientNetV4-DCNF-V6.yaml"
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
    print(f"  ✅ Full EfficientNetV4 + DCNF-V6: PASSED")

    return total_params


def test_comparison():
    """Compare parameter counts across architectures."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 11: Architecture Comparison (Parameter Counts)")
    print("=" * 70)

    configs = {
        "YOLO11 (baseline)": "ultralytics/cfg/models/11/yolo11.yaml",
        "EfficientNetV2 (standard)": "ultralytics/cfg/models/11/yolo11-EffecientNetV2/yolo11-EfficientNetV2.yaml",
        "EfficientNetV2 + DCNF-V6": "ultralytics/cfg/models/11/yolo11-EffecientNetV2/yolo11-EfficientNetV2_C2k3-DCNF-v6.yaml",
        "EfficientNetV4 (standard)": "ultralytics/cfg/models/11/yolo11-EfficientNetV4/yolo11-EfficientNetV4.yaml",
        "EfficientNetV4 + DCNF-V6": "ultralytics/cfg/models/11/yolo11-EfficientNetV4/yolo11-EfficientNetV4-DCNF-V6.yaml",
    }

    results = {}
    for name, yaml_path in configs.items():
        try:
            model = YOLO(yaml_path)
            params = count_all_params(model.model)
            results[name] = params
            print(f"  {name:35s}: {params:>12,} params")
        except Exception as e:
            print(f"  {name:35s}: FAILED ({e})")

    print()
    if "YOLO11 (baseline)" in results:
        baseline = results["YOLO11 (baseline)"]
        for name, params in results.items():
            if name != "YOLO11 (baseline)":
                diff = (params / baseline - 1) * 100
                print(f"  {name} vs baseline: {diff:+.1f}%")

    print("  ✅ Comparison: DONE")


if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║  YOLO11-EfficientNetV4 Architecture Validation Test Suite          ║")
    print("║  CASE + MKDWConv + LayerScale + Cosine DropPath                    ║")
    print("╚" + "═" * 68 + "╝")

    start = time.time()

    # Run all tests
    test_layer_scale()
    test_case()
    test_mkdwconv()
    test_fused_mbconv_v4_block()
    test_mbconv_v4_block()
    test_fused_mbconv_v4_stage()
    test_mbconv_v4_stage()
    test_gradient_flow()
    test_full_model_v4()
    test_full_model_v4_dcnf()
    test_comparison()

    elapsed = time.time() - start

    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║  ALL TESTS PASSED ✅  ({elapsed:.1f}s)                                     ║")
    print("╚" + "═" * 68 + "╝")
