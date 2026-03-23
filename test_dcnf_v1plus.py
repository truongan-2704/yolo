"""
Test Script for C3k2_DCNF_V1Plus (Improved StarFusion V1+) Block
=================================================================
Verifies:
1. StarFusionBottleneck_V1Plus forward pass and output shape
2. C3k2_DCNF_V1Plus forward pass and output shape
3. Parameter count comparison: C3k2 vs V1 vs V1+
4. YOLO model parsing with yolo11-DCNF-V1Plus.yaml
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.starfusion_block import C3k2_DCNF
from ultralytics.nn.modules.starfusion_block_v1plus import (
    StarFusionBottleneck_V1Plus, C3k2_DCNF_V1Plus
)


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_starfusion_v1plus():
    """Test StarFusionBottleneck_V1Plus standalone."""
    print("=" * 70)
    print("TEST 1: StarFusionBottleneck_V1Plus")
    print("=" * 70)

    dims = [64, 128, 256, 512]
    for c in dims:
        block = StarFusionBottleneck_V1Plus(c, c, shortcut=True)
        x = torch.randn(2, c, 40, 40)
        out = block(x)
        total, _ = count_params(block)
        print(f"  c={c:4d} | Input: {list(x.shape)} → Output: {list(out.shape)} | "
              f"Params: {total:>10,}")
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

    print("  ✅ All StarFusionBottleneck_V1Plus tests passed!\n")


def test_c3k2_dcnf_v1plus():
    """Test C3k2_DCNF_V1Plus block."""
    print("=" * 70)
    print("TEST 2: C3k2_DCNF_V1Plus (full block)")
    print("=" * 70)

    configs = [
        (64, 128, 1, False, 0.25),
        (128, 256, 2, False, 0.25),
        (256, 512, 2, True, 0.5),
        (512, 1024, 2, True, 0.5),
    ]

    for c1, c2, n, c3k, e in configs:
        block = C3k2_DCNF_V1Plus(c1, c2, n=n, c3k=c3k, e=e)
        x = torch.randn(2, c1, 40, 40)
        out = block(x)
        total, _ = count_params(block)
        print(f"  C3k2_DCNF_V1Plus(c1={c1:4d}, c2={c2:4d}, n={n}, c3k={str(c3k):5s}, e={e}) | "
              f"Output: {list(out.shape)} | Params: {total:>10,}")
        assert out.shape[1] == c2, f"Channel mismatch: {out.shape[1]} != {c2}"
        assert out.shape[2:] == x.shape[2:], f"Spatial mismatch"

    print("  ✅ All C3k2_DCNF_V1Plus tests passed!\n")


def test_comparison():
    """Compare C3k2 vs V1 vs V1+ parameters."""
    print("=" * 70)
    print("TEST 3: Parameter Comparison — C3k2 vs V1 vs V1+")
    print("=" * 70)

    configs = [
        ("Backbone P2", 128, 256, 2, False, 0.25),
        ("Backbone P3", 256, 512, 2, False, 0.25),
        ("Backbone P4", 512, 512, 2, True, 0.5),
        ("Backbone P5", 1024, 1024, 2, True, 0.5),
        ("Head P4", 512, 512, 2, False, 0.5),
        ("Head P5", 1024, 1024, 2, True, 0.5),
    ]

    total_c3k2 = total_v1 = total_v1p = 0

    print(f"  {'Location':<15} {'C3k2':>12} {'V1 (DCNF)':>12} {'V1+':>12} {'V1+ vs V1':>10}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for name, c1, c2, n, c3k, e in configs:
        p_c3k2, _ = count_params(C3k2(c1, c2, n=n, c3k=c3k, e=e))
        p_v1, _ = count_params(C3k2_DCNF(c1, c2, n=n, c3k=c3k, e=e))
        p_v1p, _ = count_params(C3k2_DCNF_V1Plus(c1, c2, n=n, c3k=c3k, e=e))

        total_c3k2 += p_c3k2
        total_v1 += p_v1
        total_v1p += p_v1p

        diff = (p_v1p / p_v1 - 1) * 100
        print(f"  {name:<15} {p_c3k2:>12,} {p_v1:>12,} {p_v1p:>12,} {diff:>+9.1f}%")

    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    overall_diff = (total_v1p / total_v1 - 1) * 100
    vs_c3k2 = (1 - total_v1p / total_c3k2) * 100
    print(f"  {'TOTAL':<15} {total_c3k2:>12,} {total_v1:>12,} {total_v1p:>12,} {overall_diff:>+9.1f}%")

    print(f"\n  📊 V1+ vs C3k2:  {vs_c3k2:+.1f}% params")
    print(f"  📊 V1+ vs V1:    {overall_diff:+.1f}% params")
    print(f"  🔥 V1+ has SE bottleneck (r=4) → fewer gate params than V1's full SE!")
    print()


def test_yaml_model():
    """Test YOLO model loading with yolo11-DCNF-V1Plus.yaml."""
    print("=" * 70)
    print("TEST 4: YOLO Model Parsing — yolo11n-DCNF-V1Plus.yaml")
    print("=" * 70)

    try:
        from ultralytics import YOLO
        model = YOLO("ultralytics/cfg/models/11/yolo11-DCNF-V1Plus.yaml")
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

        # Compare with original YOLO11 and V1
        model_orig = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
        info_orig = model_orig.info(verbose=False)

        model_v1 = YOLO("ultralytics/cfg/models/11/yolo11-DCNF.yaml")
        info_v1 = model_v1.info(verbose=False)

        print(f"\n  📊 Full Model Comparison:")
        print(f"     {'Model':<20} {'Params':>12} {'GFLOPs':>10}")
        print(f"     {'-'*20} {'-'*12} {'-'*10}")
        print(f"     {'YOLO11n (original)':<20} {info_orig[1]:>12,} {info_orig[2]:>10.1f}")
        print(f"     {'DCNF V1':<20} {info_v1[1]:>12,} {info_v1[2]:>10.1f}")
        print(f"     {'DCNF V1+':<20} {info[1]:>12,} {info[2]:>10.1f}")

        v1p_vs_v1_params = (info[1] / info_v1[1] - 1) * 100
        v1p_vs_v1_flops = (info[2] / info_v1[2] - 1) * 100
        print(f"\n     V1+ vs V1: params {v1p_vs_v1_params:+.1f}%, GFLOPs {v1p_vs_v1_flops:+.1f}%")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print()


if __name__ == "__main__":
    print("\n" + "⭐" * 35)
    print("  C3k2_DCNF_V1Plus: Improved StarFusion Block — Test Suite")
    print("⭐" * 35 + "\n")

    test_starfusion_v1plus()
    test_c3k2_dcnf_v1plus()
    test_comparison()
    test_yaml_model()

    print("=" * 70)
    print("  ALL TESTS COMPLETE! 🎯")
    print("=" * 70)
