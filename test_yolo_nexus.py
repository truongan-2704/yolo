"""
Test script for YOLO-Nexus architecture validation.

Validates:
1. Individual module shapes (OmniDirConv, NormRatioGate, PolarizedRefine)
2. NexusBottleneck forward pass
3. C3k2_Nexus and NexusCSP containers
4. Full YOLO-Nexus model build from YAML
5. Full EfficientNetV2-Nexus hybrid model build
6. Parameter count comparison with baseline YOLO11

Usage:
    python test_yolo_nexus.py
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


def test_omnidir_conv():
    """Test OmniDirConv module with various channel counts."""
    from ultralytics.nn.modules.nexus_blocks import OmniDirConv

    print("\n" + "=" * 70)
    print("TEST 1: OmniDirConv — Omnidirectional Depthwise Decomposition")
    print("=" * 70)

    for c in [32, 64, 128, 256]:
        x = torch.randn(1, c, 40, 40)
        m = OmniDirConv(c, k_strip=5, dilation=2)
        y = m(x)
        params = count_params(m)
        print(f"  c={c:4d}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    # Test with larger strip kernel (c3k=True → strip=7, dilation=3)
    x = torch.randn(1, 128, 40, 40)
    m = OmniDirConv(128, k_strip=7, dilation=3)
    y = m(x)
    print(f"  c=128 (large): input={list(x.shape)} → output={list(y.shape)}, params={count_params(m):,}")
    assert y.shape == x.shape

    print("  ✅ OmniDirConv: ALL TESTS PASSED")


def test_norm_ratio_gate():
    """Test NormRatioGate module."""
    from ultralytics.nn.modules.nexus_blocks import NormRatioGate

    print("\n" + "=" * 70)
    print("TEST 2: NormRatioGate — L1/L∞ Sparsity-Aware Channel Attention")
    print("=" * 70)

    for c in [32, 64, 128, 256]:
        x = torch.randn(1, c, 40, 40)
        m = NormRatioGate(c, reduction=8)
        y = m(x)
        params = count_params(m)
        print(f"  c={c:4d}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    # Verify sparsity ratio is computed (check beta is trainable)
    m = NormRatioGate(64)
    assert hasattr(m, 'beta'), "NormRatioGate should have learnable beta"
    assert m.beta.requires_grad, "beta should be trainable"

    print("  ✅ NormRatioGate: ALL TESTS PASSED")


def test_polarized_refine():
    """Test PolarizedRefine module."""
    from ultralytics.nn.modules.nexus_blocks import PolarizedRefine

    print("\n" + "=" * 70)
    print("TEST 3: PolarizedRefine — ON/OFF Push-Pull Contrast Enhancement")
    print("=" * 70)

    for c in [32, 64, 128, 256]:
        x = torch.randn(1, c, 40, 40)
        m = PolarizedRefine(c)
        y = m(x)
        params = count_params(m)
        print(f"  c={c:4d}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    # Verify ON/OFF decomposition works correctly
    x = torch.randn(1, 64, 20, 20)
    m = PolarizedRefine(64)
    m.eval()
    with torch.no_grad():
        y = m(x)
    # Output should be different from input (contrast enhanced)
    assert not torch.allclose(x, y, atol=1e-5), "Output should differ from input"

    print("  ✅ PolarizedRefine: ALL TESTS PASSED")


def test_nexus_bottleneck():
    """Test NexusBottleneck module."""
    from ultralytics.nn.modules.nexus_blocks import NexusBottleneck

    print("\n" + "=" * 70)
    print("TEST 4: NexusBottleneck — Core Building Block")
    print("=" * 70)

    for c in [32, 64, 128]:
        x = torch.randn(1, c, 40, 40)

        # With residual (shortcut=True, c1==c2)
        m = NexusBottleneck(c, c, shortcut=True)
        y = m(x)
        params = count_params(m)
        print(f"  c={c:4d} (res): input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape

        # Without residual (shortcut=False)
        m2 = NexusBottleneck(c, c, shortcut=False)
        y2 = m2(x)
        print(f"  c={c:4d} (nores): input={list(x.shape)} → output={list(y2.shape)}, params={count_params(m2):,}")
        assert y2.shape == x.shape

    # Channel change (c1 != c2, no residual)
    x = torch.randn(1, 64, 40, 40)
    m = NexusBottleneck(64, 128, shortcut=True)
    y = m(x)
    print(f"  64→128:     input={list(x.shape)} → output={list(y.shape)}, params={count_params(m):,}")
    assert y.shape == (1, 128, 40, 40)

    print("  ✅ NexusBottleneck: ALL TESTS PASSED")


def test_c3k2_nexus():
    """Test C3k2_Nexus container module."""
    from ultralytics.nn.modules.nexus_blocks import C3k2_Nexus

    print("\n" + "=" * 70)
    print("TEST 5: C3k2_Nexus — C2f Container (Backbone)")
    print("=" * 70)

    configs = [
        (64, 64, 1, False, "64→64, n=1, c3k=False"),
        (128, 128, 2, False, "128→128, n=2, c3k=False"),
        (256, 256, 2, True, "256→256, n=2, c3k=True"),
        (512, 1024, 2, True, "512→1024, n=2, c3k=True"),
    ]

    for c1, c2, n, c3k, desc in configs:
        x = torch.randn(1, c1, 40, 40)
        m = C3k2_Nexus(c1, c2, n=n, c3k=c3k)
        y = m(x)
        params = count_params(m)
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (1, c2, 40, 40), f"Expected (1, {c2}, 40, 40), got {y.shape}"

    print("  ✅ C3k2_Nexus: ALL TESTS PASSED")


def test_nexus_csp():
    """Test NexusCSP container module (neck)."""
    from ultralytics.nn.modules.nexus_blocks import NexusCSP

    print("\n" + "=" * 70)
    print("TEST 6: NexusCSP — C2f Container + PolarizedRefine (Neck)")
    print("=" * 70)

    configs = [
        (128, 128, 1, False, "128→128, n=1, c3k=False"),
        (256, 256, 2, False, "256→256, n=2, c3k=False"),
        (512, 512, 2, True, "512→512, n=2, c3k=True"),
        (1024, 1024, 2, True, "1024→1024, n=2, c3k=True"),
    ]

    for c1, c2, n, c3k, desc in configs:
        x = torch.randn(1, c1, 20, 20)
        m = NexusCSP(c1, c2, n=n, c3k=c3k)
        y = m(x)
        params = count_params(m)
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (1, c2, 20, 20), f"Expected (1, {c2}, 20, 20), got {y.shape}"

    print("  ✅ NexusCSP: ALL TESTS PASSED")


def test_full_model_nexus():
    """Test full YOLO-Nexus model build from YAML."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 7: Full YOLO-Nexus Model (Pure Nexus backbone + neck)")
    print("=" * 70)

    yaml_path = "ultralytics/cfg/models/11/yolo11-Nexus/yolo11-Nexus.yaml"
    print(f"  Loading: {yaml_path}")

    model = YOLO(yaml_path)
    info = model.info(verbose=False)

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

    # Count parameters
    total_params = count_all_params(model.model)
    trainable_params = count_params(model.model)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  ✅ Full YOLO-Nexus: PASSED")

    return total_params


def test_full_model_efficientnetv2_nexus():
    """Test full EfficientNetV2 + Nexus hybrid model."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 8: Full EfficientNetV2-Nexus Hybrid Model")
    print("=" * 70)

    yaml_path = "ultralytics/cfg/models/11/yolo11-Nexus/yolo11-EfficientNetV2-Nexus.yaml"
    print(f"  Loading: {yaml_path}")

    model = YOLO(yaml_path)
    info = model.info(verbose=False)

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

    # Count parameters
    total_params = count_all_params(model.model)
    trainable_params = count_params(model.model)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  ✅ Full EfficientNetV2-Nexus Hybrid: PASSED")

    return total_params


def test_comparison():
    """Compare parameter counts across architectures."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 9: Architecture Comparison (Parameter Counts)")
    print("=" * 70)

    configs = {
        "YOLO11 (baseline)": "ultralytics/cfg/models/11/yolo11.yaml",
        "YOLO-Chimera": "ultralytics/cfg/models/11/yolo11-Chimera/yolo11-Chimera.yaml",
        "YOLO-Phoenix": "ultralytics/cfg/models/11/yolo11-Phoenix/yolo11-Phoenix.yaml",
        "YOLO-EDGE": "ultralytics/cfg/models/11/yolo11-EDGE/yolo11-EDGE.yaml",
        "YOLO-Nexus": "ultralytics/cfg/models/11/yolo11-Nexus/yolo11-Nexus.yaml",
        "EfficientNetV2-Nexus": "ultralytics/cfg/models/11/yolo11-Nexus/yolo11-EfficientNetV2-Nexus.yaml",
    }

    results = {}
    for name, yaml_path in configs.items():
        try:
            model = YOLO(yaml_path)
            params = count_all_params(model.model)
            results[name] = params
            print(f"  {name:30s}: {params:>12,} params")
        except Exception as e:
            print(f"  {name:30s}: FAILED ({e})")

    if "YOLO11 (baseline)" in results and "YOLO-Nexus" in results:
        baseline = results["YOLO11 (baseline)"]
        nexus = results["YOLO-Nexus"]
        reduction = (1 - nexus / baseline) * 100
        print(f"\n  Nexus vs YOLO11: {reduction:+.1f}% parameters")

    if "YOLO-Chimera" in results and "YOLO-Nexus" in results:
        chimera = results["YOLO-Chimera"]
        nexus = results["YOLO-Nexus"]
        reduction = (1 - nexus / chimera) * 100
        print(f"  Nexus vs Chimera: {reduction:+.1f}% parameters")

    print("  ✅ Comparison: DONE")


def test_gradient_flow():
    """Test gradient flow through Nexus modules."""
    from ultralytics.nn.modules.nexus_blocks import NexusBottleneck, NexusCSP

    print("\n" + "=" * 70)
    print("TEST 10: Gradient Flow Validation")
    print("=" * 70)

    # Test gradient through NexusBottleneck
    x = torch.randn(2, 64, 20, 20, requires_grad=True)
    m = NexusBottleneck(64, 64, shortcut=True)
    y = m(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input!"
    assert x.grad.abs().sum() > 0, "Zero gradients!"
    print(f"  NexusBottleneck gradient: ✅ (grad norm = {x.grad.norm():.4f})")

    # Test gradient through NexusCSP
    x2 = torch.randn(2, 128, 20, 20, requires_grad=True)
    m2 = NexusCSP(128, 128, n=2, c3k=False)
    y2 = m2(x2)
    loss2 = y2.sum()
    loss2.backward()
    assert x2.grad is not None, "No gradient on input!"
    assert x2.grad.abs().sum() > 0, "Zero gradients!"
    print(f"  NexusCSP gradient: ✅ (grad norm = {x2.grad.norm():.4f})")

    # Check NormRatioGate beta gradient
    from ultralytics.nn.modules.nexus_blocks import NormRatioGate
    gate = NormRatioGate(64)
    x3 = torch.randn(2, 64, 10, 10)
    y3 = gate(x3)
    y3.sum().backward()
    assert gate.beta.grad is not None, "No gradient on beta!"
    print(f"  NormRatioGate β gradient: ✅ (β grad = {gate.beta.grad.item():.6f})")

    print("  ✅ Gradient Flow: ALL TESTS PASSED")


if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║    YOLO-Nexus Architecture Validation Test Suite                    ║")
    print("║    Directional-Spectral Detection with Polarity Intelligence        ║")
    print("╚" + "═" * 68 + "╝")

    start = time.time()

    # Run all tests
    test_omnidir_conv()
    test_norm_ratio_gate()
    test_polarized_refine()
    test_nexus_bottleneck()
    test_c3k2_nexus()
    test_nexus_csp()
    test_gradient_flow()
    test_full_model_nexus()
    test_full_model_efficientnetv2_nexus()
    test_comparison()

    elapsed = time.time() - start

    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║  ALL TESTS PASSED ✅  ({elapsed:.1f}s)                                     ║")
    print("╚" + "═" * 68 + "╝")
