"""
Test Script for C3k2_DCNF (Dilated Context-Star Fusion) Block
=============================================================
Verifies:
1. StarFusionBottleneck forward pass and output shape
2. C3k2_DCNF forward pass and output shape
3. Parameter count comparison: C3k2 vs C3k2_DCNF
4. YOLO model parsing with yolo11-DCNF.yaml
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.starfusion_block import StarFusionBottleneck, C3k2_DCNF


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_starfusion_bottleneck():
    """Test StarFusionBottleneck standalone."""
    print("=" * 70)
    print("TEST 1: StarFusionBottleneck")
    print("=" * 70)

    dims = [64, 128, 256, 512]
    for c in dims:
        block = StarFusionBottleneck(c, c, shortcut=True)
        x = torch.randn(2, c, 40, 40)
        out = block(x)
        total, _ = count_params(block)
        print(f"  c={c:4d} | Input: {list(x.shape)} → Output: {list(out.shape)} | "
              f"Params: {total:>10,}")
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

    print("  ✅ All StarFusionBottleneck tests passed!\n")


def test_c3k2_dcnf():
    """Test C3k2_DCNF block."""
    print("=" * 70)
    print("TEST 2: C3k2_DCNF (full block)")
    print("=" * 70)

    configs = [
        (64, 128, 1, False, 0.25),
        (128, 256, 2, False, 0.25),
        (256, 512, 2, True, 0.5),
        (512, 1024, 2, True, 0.5),
    ]

    for c1, c2, n, c3k, e in configs:
        block = C3k2_DCNF(c1, c2, n=n, c3k=c3k, e=e)
        x = torch.randn(2, c1, 40, 40)
        out = block(x)
        total, _ = count_params(block)
        print(f"  C3k2_DCNF(c1={c1:4d}, c2={c2:4d}, n={n}, c3k={str(c3k):5s}, e={e}) | "
              f"Output: {list(out.shape)} | Params: {total:>10,}")
        assert out.shape[1] == c2, f"Channel mismatch: {out.shape[1]} != {c2}"
        assert out.shape[2:] == x.shape[2:], f"Spatial mismatch"

    print("  ✅ All C3k2_DCNF tests passed!\n")


def test_comparison():
    """Compare C3k2 vs C3k2_DCNF parameters."""
    print("=" * 70)
    print("TEST 3: Parameter Comparison — C3k2 vs C3k2_DCNF")
    print("=" * 70)

    configs = [
        ("Backbone P2", 128, 256, 2, False, 0.25),
        ("Backbone P3", 256, 512, 2, False, 0.25),
        ("Backbone P4", 512, 512, 2, True, 0.5),
        ("Backbone P5", 1024, 1024, 2, True, 0.5),
        ("Head P4",  512, 512, 2, False, 0.5),
        ("Head P5", 1024, 1024, 2, True, 0.5),
    ]

    total_c3k2 = 0
    total_dcnf = 0

    print(f"  {'Location':<15} {'C3k2 Params':>12} {'DCNF Params':>12} {'Reduction':>10}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10}")

    for name, c1, c2, n, c3k, e in configs:
        c3k2_block = C3k2(c1, c2, n=n, c3k=c3k, e=e)
        dcnf_block = C3k2_DCNF(c1, c2, n=n, c3k=c3k, e=e)

        p_c3k2, _ = count_params(c3k2_block)
        p_dcnf, _ = count_params(dcnf_block)
        reduction = (1 - p_dcnf / p_c3k2) * 100

        total_c3k2 += p_c3k2
        total_dcnf += p_dcnf

        print(f"  {name:<15} {p_c3k2:>12,} {p_dcnf:>12,} {reduction:>9.1f}%")

    overall = (1 - total_dcnf / total_c3k2) * 100
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'TOTAL':<15} {total_c3k2:>12,} {total_dcnf:>12,} {overall:>9.1f}%")
    print(f"\n  🔥 C3k2_DCNF is {overall:.1f}% lighter than C3k2!\n")


def test_yaml_model():
    """Test YOLO model loading with yolo11-DCNF.yaml."""
    print("=" * 70)
    print("TEST 4: YOLO Model Parsing — yolo11n-DCNF.yaml")
    print("=" * 70)

    try:
        from ultralytics import YOLO
        model = YOLO("ultralytics/cfg/models/11/yolo11-DCNF.yaml")
        print(f"  ✅ Model loaded successfully!")

        # Model info
        info = model.info(verbose=False)
        print(f"  📊 Model Info:")
        print(f"     Layers: {info[0]}")
        print(f"     Parameters: {info[1]:,}")
        print(f"     GFLOPs: {info[2]:.1f}")

        # Forward pass test
        x = torch.randn(1, 3, 640, 640)
        result = model.model(x)
        print(f"  ✅ Forward pass successful!")

        # Compare with original YOLO11
        model_orig = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
        info_orig = model_orig.info(verbose=False)
        param_reduction = (1 - info[1] / info_orig[1]) * 100
        flop_reduction = (1 - info[2] / info_orig[2]) * 100

        print(f"\n  📊 Comparison with Original YOLO11n:")
        print(f"     Original  — Params: {info_orig[1]:>10,}  GFLOPs: {info_orig[2]:.1f}")
        print(f"     DCNF      — Params: {info[1]:>10,}  GFLOPs: {info[2]:.1f}")
        print(f"     Reduction — Params: {param_reduction:.1f}%  GFLOPs: {flop_reduction:.1f}%")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print()


if __name__ == "__main__":
    print("\n" + "🌟" * 35)
    print("  C3k2_DCNF: Dilated Context-Star Fusion Block — Test Suite")
    print("🌟" * 35 + "\n")

    test_starfusion_bottleneck()
    test_c3k2_dcnf()
    test_comparison()
    test_yaml_model()

    print("=" * 70)
    print("  ALL TESTS COMPLETE!")
    print("=" * 70)
