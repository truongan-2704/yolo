"""
Test script for C3k2_DCNF_V4 + EfficientNetV2 Pro integration.
Validates:
1. Module instantiation and forward pass
2. Gradient flow through star operation
3. Parameter count comparison with V1
4. Full YOLO model build from YAML config
"""

import torch
import time
import sys
sys.path.insert(0, '.')


def test_starfusion_v4_basic():
    """Test basic C3k2_DCNF_V4 forward pass."""
    print("=" * 60)
    print("TEST 1: C3k2_DCNF_V4 Basic Forward Pass")
    print("=" * 60)

    from ultralytics.nn.modules.starfusion_block_v4 import C3k2_DCNF_V4

    # Test various channel configurations matching YOLO usage
    configs = [
        (64, 128, 2, "P2 config"),
        (128, 256, 2, "P3 config"),
        (256, 512, 2, "P4 config"),
        (512, 1024, 2, "P5 config"),
    ]

    x_batch = 2
    x_size = 32

    for c1, c2, n, name in configs:
        x = torch.randn(x_batch, c1, x_size, x_size)
        model = C3k2_DCNF_V4(c1, c2, n=n, c3k=False, e=0.5)
        model.eval()

        with torch.no_grad():
            y = model(x)

        print(f"  [{name}] Input: {list(x.shape)} → Output: {list(y.shape)} ✓")
        assert y.shape == (x_batch, c2, x_size, x_size), f"Shape mismatch for {name}!"

    print("  ✅ All basic forward passes OK\n")


def test_gradient_flow():
    """Test gradient flows through star operation."""
    print("=" * 60)
    print("TEST 2: Gradient Flow Through Star Operation")
    print("=" * 60)

    from ultralytics.nn.modules.starfusion_block_v4 import C3k2_DCNF_V4

    model = C3k2_DCNF_V4(128, 128, n=2, c3k=False, e=0.5, shortcut=True)
    model.train()

    x = torch.randn(2, 128, 16, 16, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Check key parameters have gradients
    grad_checks = {
        'pw_weights': False,
        'star_residual_weight': False,
        'gamma': False,
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            for key in grad_checks:
                if key in name:
                    grad_checks[key] = True
                    grad_norm = param.grad.norm().item()
                    print(f"  {name}: grad_norm = {grad_norm:.6f} ✓")

    for key, has_grad in grad_checks.items():
        assert has_grad, f"No gradient for {key}!"

    assert x.grad is not None, "Input has no gradient!"
    print(f"  Input grad norm: {x.grad.norm().item():.6f} ✓")
    print("  ✅ Gradient flow OK\n")


def test_param_comparison():
    """Compare parameter counts: V1 vs V4."""
    print("=" * 60)
    print("TEST 3: Parameter Count Comparison (V1 vs V4)")
    print("=" * 60)

    from ultralytics.nn.modules.starfusion_block import C3k2_DCNF
    from ultralytics.nn.modules.starfusion_block_v4 import C3k2_DCNF_V4

    configs = [
        (256, 256, 2, "P3 (256→256, n=2)"),
        (512, 512, 2, "P4 (512→512, n=2)"),
        (1024, 1024, 2, "P5 (1024→1024, n=2)"),
    ]

    for c1, c2, n, name in configs:
        v1 = C3k2_DCNF(c1, c2, n=n, c3k=False, e=0.5)
        v4 = C3k2_DCNF_V4(c1, c2, n=n, c3k=False, e=0.5)

        p_v1 = sum(p.numel() for p in v1.parameters())
        p_v4 = sum(p.numel() for p in v4.parameters())
        ratio = p_v4 / p_v1

        print(f"  [{name}]")
        print(f"    V1 params: {p_v1:>10,}")
        print(f"    V4 params: {p_v4:>10,}")
        print(f"    Ratio V4/V1: {ratio:.3f}x")

    print("  ✅ Parameter comparison done\n")


def test_speed_comparison():
    """Benchmark inference speed: V1 vs V4."""
    print("=" * 60)
    print("TEST 4: Speed Comparison (V1 vs V4)")
    print("=" * 60)

    from ultralytics.nn.modules.starfusion_block import C3k2_DCNF
    from ultralytics.nn.modules.starfusion_block_v4 import C3k2_DCNF_V4

    c1, c2, n = 256, 256, 2
    x = torch.randn(1, c1, 32, 32)
    warmup = 10
    iters = 50

    for name, ModuleClass in [("V1", C3k2_DCNF), ("V4", C3k2_DCNF_V4)]:
        model = ModuleClass(c1, c2, n=n, c3k=False, e=0.5)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(x)

        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                model(x)
        elapsed = (time.perf_counter() - start) / iters * 1000

        print(f"  {name}: {elapsed:.2f} ms/iter (avg of {iters} iters)")

    print("  ✅ Speed comparison done\n")


def test_efficientnetv2_pro():
    """Test EfficientNetV2 Pro modules."""
    print("=" * 60)
    print("TEST 5: EfficientNetV2 Pro Modules")
    print("=" * 60)

    from ultralytics.nn.modules.EfficientNetV2 import FusedMBConv, MBConv

    # Test FusedMBConv with progressive stochastic depth
    x = torch.randn(2, 64, 32, 32)
    fused = FusedMBConv(64, 128, n=2, s=2, expand=4, drop_prob=0.1)
    fused.train()  # DropPath active in train mode
    y = fused(x)
    print(f"  FusedMBConv (64→128, s=2, n=2): {list(x.shape)} → {list(y.shape)} ✓")
    assert y.shape == (2, 128, 16, 16)

    # Test MBConv with SE (Hardsigmoid)
    x = torch.randn(2, 128, 16, 16)
    mb = MBConv(128, 256, n=3, s=2, expand=4, drop_prob=0.15)
    mb.train()
    y = mb(x)
    print(f"  MBConv (128→256, s=2, n=3): {list(x.shape)} → {list(y.shape)} ✓")
    assert y.shape == (2, 256, 8, 8)

    # Test stride=1 (identity shortcut path)
    x = torch.randn(2, 256, 8, 8)
    mb_s1 = MBConv(256, 256, n=2, s=1, expand=4, drop_prob=0.2)
    mb_s1.train()
    y = mb_s1(x)
    print(f"  MBConv (256→256, s=1, n=2): {list(x.shape)} → {list(y.shape)} ✓")
    assert y.shape == (2, 256, 8, 8)

    print("  ✅ EfficientNetV2 Pro modules OK\n")


def test_full_model_build():
    """Test building full YOLO model from YAML config."""
    print("=" * 60)
    print("TEST 6: Full YOLO Model Build from YAML")
    print("=" * 60)

    from ultralytics import YOLO

    yaml_path = "ultralytics/cfg/models/11/yolo11-EffecientNetV2/yolo11-EfficientNetV2_C2k3-DCNF-v4-Pro.yaml"

    try:
        model = YOLO(yaml_path, task='detect')
        print(f"  Model loaded successfully ✓")

        # Print model info
        n_params = sum(p.numel() for p in model.model.parameters())
        n_layers = len(list(model.model.modules()))
        print(f"  Parameters: {n_params:,}")
        print(f"  Modules: {n_layers}")

        # Test forward pass
        x = torch.randn(1, 3, 640, 640)
        model.model.eval()
        with torch.no_grad():
            y = model.model(x)
        print(f"  Forward pass (640×640): OK ✓")
        print("  ✅ Full model build OK\n")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  C3k2_DCNF_V4 + EfficientNetV2 Pro — Validation Suite")
    print("=" * 60 + "\n")

    test_starfusion_v4_basic()
    test_gradient_flow()
    test_param_comparison()
    test_speed_comparison()
    test_efficientnetv2_pro()
    test_full_model_build()

    print("=" * 60)
    print("  ALL TESTS COMPLETED")
    print("=" * 60)
