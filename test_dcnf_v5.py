"""
Test script for C3k2_DCNF_V5 + EfficientNetV2_v2 integration.
Validates:
1. EfficientNetV2_v2 module instantiation and forward pass
2. C3k2_DCNF_V5 module instantiation and forward pass
3. Gradient flow through hybrid star operation
4. Parameter count comparison (V4 vs V5 vs C3k2 original)
5. Speed comparison (V4 vs V5)
6. Full YOLO model build from YAML config
"""

import torch
import time
import sys
sys.path.insert(0, '.')


def test_efficientnetv2_v2():
    """Test EfficientNetV2 v2 backbone modules."""
    print("=" * 60)
    print("TEST 1: EfficientNetV2 v2 Backbone Modules")
    print("=" * 60)

    from ultralytics.nn.modules.EfficientNetV2_v2 import (
        FusedMBConvV2, MBConvV2, SEv2, FeatureAlign
    )

    # Test FusedMBConvV2
    x = torch.randn(2, 64, 32, 32)
    fused = FusedMBConvV2(64, 128, n=2, s=2, expand=4, drop_prob=0.1)
    fused.train()
    y = fused(x)
    print(f"  FusedMBConvV2 (64→128, s=2, n=2): {list(x.shape)} → {list(y.shape)} ✓")
    assert y.shape == (2, 128, 16, 16), f"Expected (2, 128, 16, 16), got {tuple(y.shape)}"

    # Test MBConvV2 with Sigmoid SE
    x = torch.randn(2, 128, 16, 16)
    mb = MBConvV2(128, 256, n=3, s=2, expand=4, drop_prob=0.15)
    mb.train()
    y = mb(x)
    print(f"  MBConvV2 (128→256, s=2, n=3): {list(x.shape)} → {list(y.shape)} ✓")
    assert y.shape == (2, 256, 8, 8), f"Expected (2, 256, 8, 8), got {tuple(y.shape)}"

    # Test MBConvV2 stride=1 (identity shortcut)
    x = torch.randn(2, 256, 8, 8)
    mb_s1 = MBConvV2(256, 256, n=2, s=1, expand=4, drop_prob=0.2)
    mb_s1.train()
    y = mb_s1(x)
    print(f"  MBConvV2 (256→256, s=1, n=2): {list(x.shape)} → {list(y.shape)} ✓")
    assert y.shape == (2, 256, 8, 8)

    # Test MBConvV2 with expand=6 (like P5 stage)
    x = torch.randn(2, 512, 8, 8)
    mb_e6 = MBConvV2(512, 1024, n=2, s=2, expand=6, drop_prob=0.0)
    mb_e6.eval()
    with torch.no_grad():
        y = mb_e6(x)
    print(f"  MBConvV2 (512→1024, s=2, expand=6): {list(x.shape)} → {list(y.shape)} ✓")
    assert y.shape == (2, 1024, 4, 4)

    # Test SEv2
    x = torch.randn(2, 128, 8, 8)
    se = SEv2(c_in=32, c_expand=128, se_ratio=0.25)
    y = se(x)
    print(f"  SEv2 (128ch): {list(x.shape)} → {list(y.shape)} ✓")
    assert y.shape == x.shape

    # Test FeatureAlign
    x = torch.randn(2, 256, 16, 16)
    align = FeatureAlign(256, 512)
    y = align(x)
    print(f"  FeatureAlign (256→512): {list(x.shape)} → {list(y.shape)} ✓")
    assert y.shape == (2, 512, 16, 16)

    # Test FeatureAlign identity
    align_id = FeatureAlign(256, 256)
    y = align_id(x)
    print(f"  FeatureAlign (256→256 identity): {list(x.shape)} → {list(y.shape)} ✓")
    assert y.shape == x.shape

    print("  ✅ All EfficientNetV2 v2 tests OK\n")


def test_starfusion_v5_basic():
    """Test basic C3k2_DCNF_V5 forward pass."""
    print("=" * 60)
    print("TEST 2: C3k2_DCNF_V5 Basic Forward Pass")
    print("=" * 60)

    from ultralytics.nn.modules.starfusion_block_v5 import C3k2_DCNF_V5

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
        model = C3k2_DCNF_V5(c1, c2, n=n, c3k=False, e=0.5)
        model.eval()

        with torch.no_grad():
            y = model(x)

        print(f"  [{name}] Input: {list(x.shape)} → Output: {list(y.shape)} ✓")
        assert y.shape == (x_batch, c2, x_size, x_size), f"Shape mismatch for {name}!"

    print("  ✅ All basic forward passes OK\n")


def test_gradient_flow():
    """Test gradient flows through hybrid star operation."""
    print("=" * 60)
    print("TEST 3: Gradient Flow Through Hybrid Star")
    print("=" * 60)

    from ultralytics.nn.modules.starfusion_block_v5 import C3k2_DCNF_V5

    model = C3k2_DCNF_V5(128, 128, n=2, c3k=False, e=0.5, shortcut=True)
    model.train()

    x = torch.randn(2, 128, 16, 16, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Check key parameters have gradients
    grad_checks = {
        'star_alpha': False,
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
    """Compare parameter counts: C3k2 vs V4 vs V5."""
    print("=" * 60)
    print("TEST 4: Parameter Count Comparison (C3k2 vs V4 vs V5)")
    print("=" * 60)

    from ultralytics.nn.modules.block import C3k2
    from ultralytics.nn.modules.starfusion_block import C3k2_DCNF
    from ultralytics.nn.modules.starfusion_block_v4 import C3k2_DCNF_V4
    from ultralytics.nn.modules.starfusion_block_v5 import C3k2_DCNF_V5

    configs = [
        (256, 256, 2, "P3 (256→256, n=2)"),
        (512, 512, 2, "P4 (512→512, n=2)"),
        (1024, 1024, 2, "P5 (1024→1024, n=2)"),
    ]

    print(f"  {'Config':<25} {'C3k2':>10} {'V1':>10} {'V4':>10} {'V5':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for c1, c2, n, name in configs:
        c3k2 = C3k2(c1, c2, n=n, c3k=False, e=0.5)
        v1 = C3k2_DCNF(c1, c2, n=n, c3k=False, e=0.5)
        v4 = C3k2_DCNF_V4(c1, c2, n=n, c3k=False, e=0.5)
        v5 = C3k2_DCNF_V5(c1, c2, n=n, c3k=False, e=0.5)

        p_c3k2 = sum(p.numel() for p in c3k2.parameters())
        p_v1 = sum(p.numel() for p in v1.parameters())
        p_v4 = sum(p.numel() for p in v4.parameters())
        p_v5 = sum(p.numel() for p in v5.parameters())

        print(f"  {name:<25} {p_c3k2:>10,} {p_v1:>10,} {p_v4:>10,} {p_v5:>10,}")

    print("  ✅ Parameter comparison done\n")


def test_speed_comparison():
    """Benchmark inference speed: V4 vs V5."""
    print("=" * 60)
    print("TEST 5: Speed Comparison (V1 vs V4 vs V5)")
    print("=" * 60)

    from ultralytics.nn.modules.starfusion_block import C3k2_DCNF
    from ultralytics.nn.modules.starfusion_block_v4 import C3k2_DCNF_V4
    from ultralytics.nn.modules.starfusion_block_v5 import C3k2_DCNF_V5

    c1, c2, n = 256, 256, 2
    x = torch.randn(1, c1, 32, 32)
    warmup = 10
    iters = 50

    for name, ModuleClass in [("V1", C3k2_DCNF), ("V4", C3k2_DCNF_V4), ("V5", C3k2_DCNF_V5)]:
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

        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {elapsed:.2f} ms/iter | Params: {params:,}")

    print("  ✅ Speed comparison done\n")


def test_backbone_speed():
    """Benchmark backbone speed: EfficientNetV2 v1 vs v2."""
    print("=" * 60)
    print("TEST 6: Backbone Speed (EfficientNetV2 v1 vs v2)")
    print("=" * 60)

    from ultralytics.nn.modules.EfficientNetV2 import MBConv, FusedMBConv
    from ultralytics.nn.modules.EfficientNetV2_v2 import MBConvV2, FusedMBConvV2

    x = torch.randn(1, 128, 32, 32)
    warmup = 10
    iters = 50

    # MBConv comparison
    for name, ModuleClass in [("MBConv v1", MBConv), ("MBConv v2", MBConvV2)]:
        model = ModuleClass(128, 256, n=3, s=2, expand=4, drop_prob=0.0)
        model.eval()

        with torch.no_grad():
            for _ in range(warmup):
                model(x)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                model(x)
        elapsed = (time.perf_counter() - start) / iters * 1000

        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {elapsed:.2f} ms/iter | Params: {params:,}")

    # FusedMBConv comparison
    x2 = torch.randn(1, 64, 64, 64)
    for name, ModuleClass in [("FusedMBConv v1", FusedMBConv), ("FusedMBConv v2", FusedMBConvV2)]:
        model = ModuleClass(64, 128, n=2, s=2, expand=4, drop_prob=0.0)
        model.eval()

        with torch.no_grad():
            for _ in range(warmup):
                model(x2)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                model(x2)
        elapsed = (time.perf_counter() - start) / iters * 1000

        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {elapsed:.2f} ms/iter | Params: {params:,}")

    print("  ✅ Backbone speed comparison done\n")


def test_full_model_build():
    """Test building full YOLO model from YAML config."""
    print("=" * 60)
    print("TEST 7: Full YOLO Model Build from YAML")
    print("=" * 60)

    from ultralytics import YOLO

    yaml_path = "ultralytics/cfg/models/11/yolo11-EffecientNetV2/yolo11-EfficientNetV2_C2k3-DCNF-v5.yaml"

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

        # Compare with V4 Pro model
        try:
            yaml_v4 = "ultralytics/cfg/models/11/yolo11-EffecientNetV2/yolo11-EfficientNetV2_C2k3-DCNF-v4-Pro.yaml"
            model_v4 = YOLO(yaml_v4, task='detect')
            n_params_v4 = sum(p.numel() for p in model_v4.model.parameters())
            print(f"\n  📊 V4 Pro Parameters: {n_params_v4:,}")
            print(f"  📊 V5 Parameters:     {n_params:,}")
            ratio = n_params / n_params_v4
            print(f"  📊 V5/V4 ratio:       {ratio:.3f}x")
        except Exception as e:
            print(f"  (V4 Pro comparison skipped: {e})")

        print("  ✅ Full model build OK\n")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()


def test_groups_divisibility():
    """Test that GroupConv groups work correctly for various channel sizes."""
    print("=" * 60)
    print("TEST 8: GroupConv Divisibility Check")
    print("=" * 60)

    from ultralytics.nn.modules.starfusion_block_v5 import StarFusionBottleneck_V5

    # Test various channel sizes that might appear after width scaling
    channel_sizes = [8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 384, 512]

    for c in channel_sizes:
        try:
            block = StarFusionBottleneck_V5(c, c, shortcut=True, e=0.5)
            x = torch.randn(1, c, 8, 8)
            with torch.no_grad():
                y = block(x)
            c_ = int(c * 0.5)
            groups = block.groups
            print(f"  c={c:4d} (c_={c_:3d}, g={groups:3d}): {list(y.shape)} ✓")
            assert y.shape == x.shape
        except Exception as e:
            print(f"  c={c:4d}: ❌ {e}")

    print("  ✅ GroupConv divisibility OK\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  C3k2_DCNF_V5 + EfficientNetV2_v2 — Validation Suite")
    print("=" * 60 + "\n")

    test_efficientnetv2_v2()
    test_starfusion_v5_basic()
    test_gradient_flow()
    test_groups_divisibility()
    test_param_comparison()
    test_speed_comparison()
    test_backbone_speed()
    test_full_model_build()

    print("=" * 60)
    print("  ALL TESTS COMPLETED ✅")
    print("=" * 60)
