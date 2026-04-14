"""
Test script for YOLO-NexusFast architecture validation and speed comparison.

Validates:
1. Individual Fast modules (FastOmniDirConv, FastNormRatioGate, FastPolarizedRefine)
2. FastNexusBottleneck forward pass
3. C3k2_NexusFast and NexusCSPFast containers
4. Full EfficientNetV2-NexusFast hybrid model build
5. Speed comparison: Original vs Fast variants
6. Parameter count comparison

Usage:
    python test_yolo_nexus_fast.py
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


def benchmark_module(module, x, n_warmup=10, n_iter=50, label=""):
    """Benchmark a module's forward pass speed."""
    module.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = module(x)

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = module(x)
        elapsed = (time.perf_counter() - start) / n_iter * 1000  # ms

    print(f"    {label}: {elapsed:.2f} ms/forward")
    return elapsed


def test_fast_omnidir_conv():
    """Test FastOmniDirConv module."""
    from ultralytics.nn.modules.nexus_blocks import OmniDirConv, FastOmniDirConv

    print("\n" + "=" * 70)
    print("TEST 1: FastOmniDirConv — No Channel Shuffle")
    print("=" * 70)

    for c in [32, 64, 128, 256]:
        x = torch.randn(1, c, 40, 40)
        m = FastOmniDirConv(c, k_strip=5, dilation=2)
        y = m(x)
        params = count_params(m)
        print(f"  c={c:4d}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    # Speed comparison
    print("\n  Speed comparison (c=128, 40×40):")
    x = torch.randn(1, 128, 40, 40)
    orig = OmniDirConv(128)
    fast = FastOmniDirConv(128)
    t_orig = benchmark_module(orig, x, label="OmniDirConv (original)")
    t_fast = benchmark_module(fast, x, label="FastOmniDirConv (fast)  ")
    speedup = t_orig / t_fast
    print(f"    Speedup: {speedup:.2f}×")

    print("  ✅ FastOmniDirConv: ALL TESTS PASSED")


def test_fast_norm_ratio_gate():
    """Test FastNormRatioGate module."""
    from ultralytics.nn.modules.nexus_blocks import NormRatioGate, FastNormRatioGate

    print("\n" + "=" * 70)
    print("TEST 2: FastNormRatioGate — No x.abs()")
    print("=" * 70)

    for c in [32, 64, 128, 256]:
        x = torch.randn(1, c, 40, 40)
        m = FastNormRatioGate(c, reduction=8)
        y = m(x)
        params = count_params(m)
        print(f"  c={c:4d}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    # Verify beta is trainable
    m = FastNormRatioGate(64)
    assert hasattr(m, 'beta'), "FastNormRatioGate should have learnable beta"
    assert m.beta.requires_grad, "beta should be trainable"

    # Speed comparison
    print("\n  Speed comparison (c=256, 20×20):")
    x = torch.randn(1, 256, 20, 20)
    orig = NormRatioGate(256)
    fast = FastNormRatioGate(256)
    t_orig = benchmark_module(orig, x, label="NormRatioGate (original)")
    t_fast = benchmark_module(fast, x, label="FastNormRatioGate (fast) ")
    speedup = t_orig / t_fast
    print(f"    Speedup: {speedup:.2f}×")

    print("  ✅ FastNormRatioGate: ALL TESTS PASSED")


def test_fast_polarized_refine():
    """Test FastPolarizedRefine module."""
    from ultralytics.nn.modules.nexus_blocks import PolarizedRefine, FastPolarizedRefine

    print("\n" + "=" * 70)
    print("TEST 3: FastPolarizedRefine — Single-Path Spatial Attention")
    print("=" * 70)

    for c in [32, 64, 128, 256]:
        x = torch.randn(1, c, 40, 40)
        m = FastPolarizedRefine(c)
        y = m(x)
        params = count_params(m)
        print(f"  c={c:4d}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    # Speed comparison
    print("\n  Speed comparison (c=256, 20×20):")
    x = torch.randn(1, 256, 20, 20)
    orig = PolarizedRefine(256)
    fast = FastPolarizedRefine(256)
    t_orig = benchmark_module(orig, x, label="PolarizedRefine (original)")
    t_fast = benchmark_module(fast, x, label="FastPolarizedRefine (fast) ")
    speedup = t_orig / t_fast
    print(f"    Speedup: {speedup:.2f}×")

    # Param comparison
    p_orig = count_params(orig)
    p_fast = count_params(fast)
    reduction = (1 - p_fast / p_orig) * 100
    print(f"    Params: {p_orig:,} → {p_fast:,} ({reduction:.1f}% reduction)")

    print("  ✅ FastPolarizedRefine: ALL TESTS PASSED")


def test_fast_nexus_bottleneck():
    """Test FastNexusBottleneck module."""
    from ultralytics.nn.modules.nexus_blocks import NexusBottleneck, FastNexusBottleneck

    print("\n" + "=" * 70)
    print("TEST 4: FastNexusBottleneck — No Channel Expansion")
    print("=" * 70)

    for c in [32, 64, 128]:
        x = torch.randn(1, c, 40, 40)

        # With residual
        m = FastNexusBottleneck(c, c, shortcut=True)
        y = m(x)
        params = count_params(m)
        print(f"  c={c:4d} (res): input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape

        # Without residual
        m2 = FastNexusBottleneck(c, c, shortcut=False)
        y2 = m2(x)
        print(f"  c={c:4d} (nores): input={list(x.shape)} → output={list(y2.shape)}, params={count_params(m2):,}")
        assert y2.shape == x.shape

    # Channel change
    x = torch.randn(1, 64, 40, 40)
    m = FastNexusBottleneck(64, 128, shortcut=True)
    y = m(x)
    print(f"  64→128:     input={list(x.shape)} → output={list(y.shape)}, params={count_params(m):,}")
    assert y.shape == (1, 128, 40, 40)

    # Speed comparison
    print("\n  Speed comparison (c=128, 40×40):")
    x = torch.randn(1, 128, 40, 40)
    orig = NexusBottleneck(128, 128)
    fast = FastNexusBottleneck(128, 128)
    t_orig = benchmark_module(orig, x, label="NexusBottleneck (original)")
    t_fast = benchmark_module(fast, x, label="FastNexusBottleneck (fast) ")
    speedup = t_orig / t_fast
    print(f"    Speedup: {speedup:.2f}×")

    p_orig = count_params(orig)
    p_fast = count_params(fast)
    reduction = (1 - p_fast / p_orig) * 100
    print(f"    Params: {p_orig:,} → {p_fast:,} ({reduction:.1f}% reduction)")

    print("  ✅ FastNexusBottleneck: ALL TESTS PASSED")


def test_c3k2_nexus_fast():
    """Test C3k2_NexusFast container module."""
    from ultralytics.nn.modules.nexus_blocks import C3k2_Nexus, C3k2_NexusFast

    print("\n" + "=" * 70)
    print("TEST 5: C3k2_NexusFast — C2f Container (Backbone)")
    print("=" * 70)

    configs = [
        (64, 64, 1, False, "64→64, n=1, c3k=False"),
        (128, 128, 2, False, "128→128, n=2, c3k=False"),
        (256, 256, 2, True, "256→256, n=2, c3k=True"),
        (512, 1024, 2, True, "512→1024, n=2, c3k=True"),
    ]

    for c1, c2, n, c3k, desc in configs:
        x = torch.randn(1, c1, 40, 40)
        m = C3k2_NexusFast(c1, c2, n=n, c3k=c3k)
        y = m(x)
        params = count_params(m)
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (1, c2, 40, 40)

    # Speed comparison
    print("\n  Speed comparison (256→256, n=2, 40×40):")
    x = torch.randn(1, 256, 40, 40)
    orig = C3k2_Nexus(256, 256, n=2)
    fast = C3k2_NexusFast(256, 256, n=2)
    t_orig = benchmark_module(orig, x, label="C3k2_Nexus (original)   ")
    t_fast = benchmark_module(fast, x, label="C3k2_NexusFast (fast)   ")
    speedup = t_orig / t_fast
    print(f"    Speedup: {speedup:.2f}×")

    print("  ✅ C3k2_NexusFast: ALL TESTS PASSED")


def test_nexus_csp_fast():
    """Test NexusCSPFast container module (neck)."""
    from ultralytics.nn.modules.nexus_blocks import NexusCSP, NexusCSPFast

    print("\n" + "=" * 70)
    print("TEST 6: NexusCSPFast — C2f + FastPolarizedRefine (Neck)")
    print("=" * 70)

    configs = [
        (128, 128, 1, False, "128→128, n=1, c3k=False"),
        (256, 256, 2, False, "256→256, n=2, c3k=False"),
        (512, 512, 2, True, "512→512, n=2, c3k=True"),
        (1024, 1024, 2, True, "1024→1024, n=2, c3k=True"),
    ]

    for c1, c2, n, c3k, desc in configs:
        x = torch.randn(1, c1, 20, 20)
        m = NexusCSPFast(c1, c2, n=n, c3k=c3k)
        y = m(x)
        params = count_params(m)
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (1, c2, 20, 20)

    # Speed comparison
    print("\n  Speed comparison (512→512, n=2, 20×20):")
    x = torch.randn(1, 512, 20, 20)
    orig = NexusCSP(512, 512, n=2)
    fast = NexusCSPFast(512, 512, n=2)
    t_orig = benchmark_module(orig, x, label="NexusCSP (original)   ")
    t_fast = benchmark_module(fast, x, label="NexusCSPFast (fast)   ")
    speedup = t_orig / t_fast
    print(f"    Speedup: {speedup:.2f}×")

    print("  ✅ NexusCSPFast: ALL TESTS PASSED")


def test_gradient_flow():
    """Test gradient flow through Fast modules."""
    from ultralytics.nn.modules.nexus_blocks import FastNexusBottleneck, NexusCSPFast

    print("\n" + "=" * 70)
    print("TEST 7: Gradient Flow Validation")
    print("=" * 70)

    # Test gradient through FastNexusBottleneck
    x = torch.randn(2, 64, 20, 20, requires_grad=True)
    m = FastNexusBottleneck(64, 64, shortcut=True)
    y = m(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input!"
    assert x.grad.abs().sum() > 0, "Zero gradients!"
    print(f"  FastNexusBottleneck gradient: ✅ (grad norm = {x.grad.norm():.4f})")

    # Test gradient through NexusCSPFast
    x2 = torch.randn(2, 128, 20, 20, requires_grad=True)
    m2 = NexusCSPFast(128, 128, n=2, c3k=False)
    y2 = m2(x2)
    loss2 = y2.sum()
    loss2.backward()
    assert x2.grad is not None, "No gradient on input!"
    assert x2.grad.abs().sum() > 0, "Zero gradients!"
    print(f"  NexusCSPFast gradient: ✅ (grad norm = {x2.grad.norm():.4f})")

    # Check FastNormRatioGate beta gradient
    from ultralytics.nn.modules.nexus_blocks import FastNormRatioGate
    gate = FastNormRatioGate(64)
    x3 = torch.randn(2, 64, 10, 10)
    y3 = gate(x3)
    y3.sum().backward()
    assert gate.beta.grad is not None, "No gradient on beta!"
    print(f"  FastNormRatioGate β gradient: ✅ (β grad = {gate.beta.grad.item():.6f})")

    print("  ✅ Gradient Flow: ALL TESTS PASSED")


def test_full_model():
    """Test full EfficientNetV2-NexusFast model build and speed."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 8: Full EfficientNetV2-NexusFast Model")
    print("=" * 70)

    # Build NexusFast model
    yaml_fast = "ultralytics/cfg/models/11/yolo11-Nexus/yolo11-EfficientNetV2-NexusFast.yaml"
    print(f"  Loading: {yaml_fast}")
    model_fast = YOLO(yaml_fast)

    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    model_fast.model.eval()
    with torch.no_grad():
        y = model_fast.model(x)

    print(f"  ✅ Model built successfully!")
    print(f"  Input: {list(x.shape)}")
    if isinstance(y, (list, tuple)):
        for i, yi in enumerate(y):
            if isinstance(yi, torch.Tensor):
                print(f"  Output[{i}]: {list(yi.shape)}")

    total_params = count_all_params(model_fast.model)
    trainable_params = count_params(model_fast.model)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    print("  ✅ Full EfficientNetV2-NexusFast: PASSED")
    return total_params


def test_speed_comparison():
    """Compare inference speed between original and fast models."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 9: Full Model Speed Comparison (Original vs Fast)")
    print("=" * 70)

    yaml_orig = "ultralytics/cfg/models/11/yolo11-Nexus/yolo11-EfficientNetV2-Nexus.yaml"
    yaml_fast = "ultralytics/cfg/models/11/yolo11-Nexus/yolo11-EfficientNetV2-NexusFast.yaml"

    model_orig = YOLO(yaml_orig)
    model_fast = YOLO(yaml_fast)

    model_orig.model.eval()
    model_fast.model.eval()

    x = torch.randn(1, 3, 640, 640)

    # Warmup
    n_warmup = 5
    n_iter = 20
    print(f"  Benchmarking (warmup={n_warmup}, iterations={n_iter})...")

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_orig.model(x)
            _ = model_fast.model(x)

        # Benchmark original
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = model_orig.model(x)
        t_orig = (time.perf_counter() - start) / n_iter * 1000

        # Benchmark fast
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = model_fast.model(x)
        t_fast = (time.perf_counter() - start) / n_iter * 1000

    speedup = t_orig / t_fast
    p_orig = count_all_params(model_orig.model)
    p_fast = count_all_params(model_fast.model)
    p_reduction = (1 - p_fast / p_orig) * 100

    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ Model                      │ Params      │ Speed (ms)  │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ EfficientNetV2-Nexus       │ {p_orig:>11,} │ {t_orig:>8.2f} ms │")
    print(f"  │ EfficientNetV2-NexusFast   │ {p_fast:>11,} │ {t_fast:>8.2f} ms │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ Improvement                │ {p_reduction:>+10.1f}% │ {speedup:>7.2f}×    │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    print(f"\n  ✅ Speed Comparison: DONE")


def test_param_comparison():
    """Compare parameter counts across architectures."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 10: Architecture Comparison (Parameter Counts)")
    print("=" * 70)

    configs = {
        "YOLO11 (baseline)": "ultralytics/cfg/models/11/yolo11.yaml",
        "EfficientNetV2-Nexus": "ultralytics/cfg/models/11/yolo11-Nexus/yolo11-EfficientNetV2-Nexus.yaml",
        "EfficientNetV2-NexusFast": "ultralytics/cfg/models/11/yolo11-Nexus/yolo11-EfficientNetV2-NexusFast.yaml",
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

    if "EfficientNetV2-Nexus" in results and "EfficientNetV2-NexusFast" in results:
        orig = results["EfficientNetV2-Nexus"]
        fast = results["EfficientNetV2-NexusFast"]
        reduction = (1 - fast / orig) * 100
        print(f"\n  NexusFast vs Nexus: {reduction:+.1f}% parameters")

    print("  ✅ Comparison: DONE")


if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║    YOLO-NexusFast Architecture Validation & Speed Test             ║")
    print("║    Speed-Optimized Directional-Spectral Detection                  ║")
    print("╚" + "═" * 68 + "╝")

    start = time.time()

    # Module tests
    test_fast_omnidir_conv()
    test_fast_norm_ratio_gate()
    test_fast_polarized_refine()
    test_fast_nexus_bottleneck()
    test_c3k2_nexus_fast()
    test_nexus_csp_fast()
    test_gradient_flow()

    # Full model tests
    test_full_model()
    test_speed_comparison()
    test_param_comparison()

    elapsed = time.time() - start

    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║  ALL TESTS PASSED ✅  ({elapsed:.1f}s)                                     ║")
    print("╚" + "═" * 68 + "╝")
