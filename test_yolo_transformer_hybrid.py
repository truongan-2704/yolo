"""
Test script for YOLO11-Transformer Hybrid architecture validation.

Validates:
1. Swin Transformer components (WindowAttention, SwinTransformerBlock, PatchMerging, SwinStage)
2. ViT Hybrid components (ViTSelfAttention, ViTBlock, ViTStage)
3. Mobile-Former components (MobileBlock, FormerBlock, bridges, MobileFormerStage)
4. Full YOLO11-Swin model build from YAML
5. Full YOLO11-ViT model build from YAML
6. Full YOLO11-MobileFormer model build from YAML
7. Parameter count comparison with baseline

Usage:
    python test_yolo_transformer_hybrid.py
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


def test_window_attention():
    """Test WindowAttention module."""
    from ultralytics.nn.modules.TransformerHybrid import WindowAttention

    print("\n" + "=" * 70)
    print("TEST 1: WindowAttention — Window-based Multi-Head Self-Attention")
    print("=" * 70)

    for dim, num_heads, ws in [(64, 2, 7), (128, 4, 7), (256, 8, 7)]:
        N = ws * ws  # tokens per window
        x = torch.randn(4, N, dim)  # (num_windows*B, N, C)
        m = WindowAttention(dim, window_size=(ws, ws), num_heads=num_heads)
        y = m(x)
        params = count_params(m)
        print(f"  dim={dim:4d}, heads={num_heads}, ws={ws}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    # Test with attention mask (shifted window)
    x = torch.randn(8, 49, 128)  # 49 = 7*7
    mask = torch.zeros(2, 49, 49)
    m = WindowAttention(128, window_size=(7, 7), num_heads=4)
    y = m(x, mask=mask)
    assert y.shape == x.shape
    print(f"  With mask: ✅")

    print("  ✅ WindowAttention: ALL TESTS PASSED")


def test_swin_block():
    """Test SwinTransformerBlock module."""
    from ultralytics.nn.modules.TransformerHybrid import SwinTransformerBlock

    print("\n" + "=" * 70)
    print("TEST 2: SwinTransformerBlock — W-MSA / SW-MSA")
    print("=" * 70)

    # W-MSA (no shift)
    x = torch.randn(2, 128, 28, 28)
    m = SwinTransformerBlock(dim=128, num_heads=4, window_size=7, shift_size=0)
    y = m(x)
    print(f"  W-MSA:  input={list(x.shape)} → output={list(y.shape)}, params={count_params(m):,}")
    assert y.shape == x.shape

    # SW-MSA (shifted window)
    m2 = SwinTransformerBlock(dim=128, num_heads=4, window_size=7, shift_size=3)
    y2 = m2(x)
    print(f"  SW-MSA: input={list(x.shape)} → output={list(y2.shape)}, params={count_params(m2):,}")
    assert y2.shape == x.shape

    # Non-divisible spatial size (tests padding)
    x3 = torch.randn(2, 128, 20, 20)  # 20 not divisible by 7
    y3 = m(x3)
    print(f"  Padding: input={list(x3.shape)} → output={list(y3.shape)}")
    assert y3.shape == x3.shape

    print("  ✅ SwinTransformerBlock: ALL TESTS PASSED")


def test_patch_merging():
    """Test PatchMerging module."""
    from ultralytics.nn.modules.TransformerHybrid import PatchMerging

    print("\n" + "=" * 70)
    print("TEST 3: PatchMerging — Spatial Downsampling (2x)")
    print("=" * 70)

    for c1, c2 in [(64, 128), (128, 256), (256, 512)]:
        x = torch.randn(2, c1, 28, 28)
        m = PatchMerging(c1, c2)
        y = m(x)
        params = count_params(m)
        print(f"  {c1}→{c2}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, 14, 14)

    # Test with odd dimensions
    x = torch.randn(2, 64, 27, 27)
    m = PatchMerging(64, 128)
    y = m(x)
    print(f"  Odd dims: input={list(x.shape)} → output={list(y.shape)}")
    assert y.shape[2] == 14 and y.shape[3] == 14  # (27+1)/2 = 14

    print("  ✅ PatchMerging: ALL TESTS PASSED")


def test_swin_stage():
    """Test SwinStage module."""
    from ultralytics.nn.modules.TransformerHybrid import SwinStage

    print("\n" + "=" * 70)
    print("TEST 4: SwinStage — Swin Transformer Stage (YOLO compatible)")
    print("=" * 70)

    configs = [
        (128, 256, 2, 2, "128→256, n=2, s=2 (PatchMerging)"),
        (256, 512, 3, 2, "256→512, n=3, s=2"),
        (512, 512, 2, 1, "512→512, n=2, s=1 (no downsample)"),
        (512, 1024, 2, 2, "512→1024, n=2, s=2"),
    ]

    for c1, c2, n, s, desc in configs:
        x = torch.randn(2, c1, 28, 28)
        m = SwinStage(c1, c2, n=n, s=s, window_size=7)
        y = m(x)
        params = count_params(m)
        expected_h = 28 // s
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, expected_h, expected_h), f"Expected ({2}, {c2}, {expected_h}, {expected_h}), got {y.shape}"

    print("  ✅ SwinStage: ALL TESTS PASSED")


def test_vit_block():
    """Test ViTBlock module."""
    from ultralytics.nn.modules.TransformerHybrid import ViTBlock

    print("\n" + "=" * 70)
    print("TEST 5: ViTBlock — Standard ViT Transformer Block")
    print("=" * 70)

    for dim, num_heads in [(64, 2), (128, 4), (256, 8)]:
        N = 100  # 10x10 spatial
        x = torch.randn(2, N, dim)
        m = ViTBlock(dim=dim, num_heads=num_heads)
        y = m(x)
        params = count_params(m)
        print(f"  dim={dim:4d}, heads={num_heads}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape

    print("  ✅ ViTBlock: ALL TESTS PASSED")


def test_vit_stage():
    """Test ViTStage module."""
    from ultralytics.nn.modules.TransformerHybrid import ViTStage

    print("\n" + "=" * 70)
    print("TEST 6: ViTStage — ViT Hybrid Stage (YOLO compatible)")
    print("=" * 70)

    configs = [
        (128, 256, 2, 2, "128→256, n=2, s=2 (conv downsample)"),
        (256, 512, 3, 2, "256→512, n=3, s=2"),
        (512, 512, 2, 1, "512→512, n=2, s=1 (no downsample)"),
        (512, 1024, 2, 2, "512→1024, n=2, s=2"),
    ]

    for c1, c2, n, s, desc in configs:
        x = torch.randn(2, c1, 20, 20)
        m = ViTStage(c1, c2, n=n, s=s)
        y = m(x)
        params = count_params(m)
        expected_h = 20 // s
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, expected_h, expected_h), f"Expected ({2}, {c2}, {expected_h}, {expected_h}), got {y.shape}"

    print("  ✅ ViTStage: ALL TESTS PASSED")


def test_mobile_former_components():
    """Test Mobile-Former individual components."""
    from ultralytics.nn.modules.TransformerHybrid import (
        MobileBlock, FormerBlock, Mobile2Former, Former2Mobile
    )

    print("\n" + "=" * 70)
    print("TEST 7: Mobile-Former Components")
    print("=" * 70)

    dim = 128

    # MobileBlock
    x = torch.randn(2, dim, 20, 20)
    mb = MobileBlock(dim, expand=4)
    y = mb(x)
    print(f"  MobileBlock: input={list(x.shape)} → output={list(y.shape)}, params={count_params(mb):,}")
    assert y.shape == x.shape

    # FormerBlock
    tokens = torch.randn(2, 6, dim)
    fb = FormerBlock(dim, num_heads=4)
    t_out = fb(tokens)
    print(f"  FormerBlock: input={list(tokens.shape)} → output={list(t_out.shape)}, params={count_params(fb):,}")
    assert t_out.shape == tokens.shape

    # Mobile2Former bridge
    m2f = Mobile2Former(dim, num_heads=4)
    t_updated = m2f(tokens, x)
    print(f"  Mobile2Former: tokens={list(tokens.shape)}, spatial={list(x.shape)} → tokens={list(t_updated.shape)}, params={count_params(m2f):,}")
    assert t_updated.shape == tokens.shape

    # Former2Mobile bridge
    f2m = Former2Mobile(dim, num_heads=4)
    x_updated = f2m(x, tokens)
    print(f"  Former2Mobile: spatial={list(x.shape)}, tokens={list(tokens.shape)} → spatial={list(x_updated.shape)}, params={count_params(f2m):,}")
    assert x_updated.shape == x.shape

    print("  ✅ Mobile-Former Components: ALL TESTS PASSED")


def test_mobile_former_stage():
    """Test MobileFormerStage module."""
    from ultralytics.nn.modules.TransformerHybrid import MobileFormerStage

    print("\n" + "=" * 70)
    print("TEST 8: MobileFormerStage — Bidirectional Conv+Transformer (YOLO compatible)")
    print("=" * 70)

    configs = [
        (128, 256, 2, 2, "128→256, n=2, s=2"),
        (256, 512, 3, 2, "256→512, n=3, s=2"),
        (512, 512, 2, 1, "512→512, n=2, s=1"),
        (512, 1024, 2, 2, "512→1024, n=2, s=2"),
    ]

    for c1, c2, n, s, desc in configs:
        x = torch.randn(2, c1, 20, 20)
        m = MobileFormerStage(c1, c2, n=n, s=s, num_tokens=6)
        y = m(x)
        params = count_params(m)
        expected_h = 20 // s
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, expected_h, expected_h), f"Expected ({2}, {c2}, {expected_h}, {expected_h}), got {y.shape}"

    print("  ✅ MobileFormerStage: ALL TESTS PASSED")


def test_gradient_flow():
    """Test gradient flow through all Transformer hybrid modules."""
    from ultralytics.nn.modules.TransformerHybrid import SwinStage, ViTStage, MobileFormerStage

    print("\n" + "=" * 70)
    print("TEST 9: Gradient Flow Validation")
    print("=" * 70)

    # SwinStage gradient
    x1 = torch.randn(2, 64, 28, 28, requires_grad=True)
    m1 = SwinStage(64, 128, n=2, s=2, window_size=7)
    y1 = m1(x1)
    y1.sum().backward()
    assert x1.grad is not None and x1.grad.abs().sum() > 0
    print(f"  SwinStage gradient: ✅ (grad norm = {x1.grad.norm():.4f})")

    # ViTStage gradient
    x2 = torch.randn(2, 64, 20, 20, requires_grad=True)
    m2 = ViTStage(64, 128, n=2, s=2)
    y2 = m2(x2)
    y2.sum().backward()
    assert x2.grad is not None and x2.grad.abs().sum() > 0
    print(f"  ViTStage gradient: ✅ (grad norm = {x2.grad.norm():.4f})")

    # MobileFormerStage gradient
    x3 = torch.randn(2, 64, 20, 20, requires_grad=True)
    m3 = MobileFormerStage(64, 128, n=2, s=2, num_tokens=6)
    y3 = m3(x3)
    y3.sum().backward()
    assert x3.grad is not None and x3.grad.abs().sum() > 0
    print(f"  MobileFormerStage gradient: ✅ (grad norm = {x3.grad.norm():.4f})")

    print("  ✅ Gradient Flow: ALL TESTS PASSED")


def test_full_model_swin():
    """Test full YOLO11-Swin model build from YAML."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 10: Full YOLO11-Swin Transformer Model")
    print("=" * 70)

    yaml_path = "ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-Swin.yaml"
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
    print(f"  ✅ Full YOLO11-Swin: PASSED")

    return total_params


def test_full_model_vit():
    """Test full YOLO11-ViT model build from YAML."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 11: Full YOLO11-ViT Hybrid Model")
    print("=" * 70)

    yaml_path = "ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-ViT.yaml"
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
    print(f"  ✅ Full YOLO11-ViT: PASSED")

    return total_params


def test_full_model_mobile_former():
    """Test full YOLO11-MobileFormer model build from YAML."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 12: Full YOLO11-MobileFormer Model")
    print("=" * 70)

    yaml_path = "ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-MobileFormer.yaml"
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
    print(f"  ✅ Full YOLO11-MobileFormer: PASSED")

    return total_params


def test_comparison():
    """Compare parameter counts across architectures."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 13: Architecture Comparison (Parameter Counts)")
    print("=" * 70)

    configs = {
        "YOLO11 (baseline)": "ultralytics/cfg/models/11/yolo11.yaml",
        "YOLO11-Swin": "ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-Swin.yaml",
        "YOLO11-ViT": "ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-ViT.yaml",
        "YOLO11-MobileFormer": "ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-MobileFormer.yaml",
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
    print("║  YOLO11-Transformer Hybrid Architecture Validation Test Suite     ║")
    print("║  Swin Transformer + ViT Hybrid + Mobile-Former                    ║")
    print("╚" + "═" * 68 + "╝")

    start = time.time()

    # Run all tests
    test_window_attention()
    test_swin_block()
    test_patch_merging()
    test_swin_stage()
    test_vit_block()
    test_vit_stage()
    test_mobile_former_components()
    test_mobile_former_stage()
    test_gradient_flow()
    test_full_model_swin()
    test_full_model_vit()
    test_full_model_mobile_former()
    test_comparison()

    elapsed = time.time() - start

    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║  ALL TESTS PASSED ✅  ({elapsed:.1f}s)                                     ║")
    print("╚" + "═" * 68 + "╝")
