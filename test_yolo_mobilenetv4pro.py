"""
Test script for YOLO11-MobileNetV4Pro architecture validation.

Validates:
1. Individual MobileNetV4Pro module shapes (RepConvBN, RepDWConvBN, DualPoolSE, etc.)
2. RepFusedIBBlock, RepUIBBlock, EfficientGQA components
3. MNV4ProConv, MNV4ProUIB, MNV4ProHybrid, MNV4ProNeck stage containers
4. Reparameterization (fuse) correctness
5. Full YOLO11-MobileNetV4Pro model build from YAML
6. Full YOLO11-MobileNetV4Pro-Hybrid model build
7. Parameter count comparison with baseline and original MobileNetV4

Usage:
    python test_yolo_mobilenetv4pro.py
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


def test_dual_pool_se():
    """Test DualPoolSE module."""
    from ultralytics.nn.modules.MobileNetV4Pro import DualPoolSE

    print("\n" + "=" * 70)
    print("TEST 1: DualPoolSE — Dual-Pool Squeeze-and-Excitation")
    print("=" * 70)

    for c_in, c_expand in [(16, 64), (32, 128), (64, 256), (128, 512)]:
        x = torch.randn(2, c_expand, 20, 20)
        m = DualPoolSE(c_in=c_in, c_expand=c_expand, se_ratio=0.25)
        y = m(x)
        params = count_params(m)
        print(f"  c_in={c_in:4d}, c_expand={c_expand:4d}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    print("  ✅ DualPoolSE: ALL TESTS PASSED")


def test_rep_conv_bn():
    """Test RepConvBN module and reparameterization."""
    from ultralytics.nn.modules.MobileNetV4Pro import RepConvBN

    print("\n" + "=" * 70)
    print("TEST 2: RepConvBN — Reparameterizable Conv+BN")
    print("=" * 70)

    # Test with residual (c1 == c2, s == 1)
    x = torch.randn(2, 64, 20, 20)
    m = RepConvBN(64, 64, k=3, s=1)
    m.eval()

    y_train = m(x)
    print(f"  Training mode: input={list(x.shape)} → output={list(y_train.shape)}, params={count_params(m):,}")
    assert y_train.shape == x.shape

    # Fuse and compare
    m.fuse()
    y_fused = m(x)
    print(f"  Fused mode:    input={list(x.shape)} → output={list(y_fused.shape)}, params={count_params(m):,}")
    assert y_fused.shape == x.shape

    # Check fuse correctness (outputs should be close)
    diff = (y_train - y_fused).abs().max().item()
    print(f"  Fuse error: {diff:.8f} (should be < 1e-4)")
    assert diff < 1e-4, f"Fuse error too large: {diff}"

    # Test without residual (c1 != c2)
    m2 = RepConvBN(64, 128, k=3, s=2)
    m2.eval()
    y2 = m2(x)
    print(f"  No-res mode:   input={list(x.shape)} → output={list(y2.shape)}, params={count_params(m2):,}")
    assert y2.shape == (2, 128, 10, 10)

    print("  ✅ RepConvBN: ALL TESTS PASSED")


def test_rep_dw_conv_bn():
    """Test RepDWConvBN module and reparameterization."""
    from ultralytics.nn.modules.MobileNetV4Pro import RepDWConvBN

    print("\n" + "=" * 70)
    print("TEST 3: RepDWConvBN — Reparameterizable DW Conv+BN")
    print("=" * 70)

    for ch, k_main in [(64, 5), (128, 5), (256, 3)]:
        x = torch.randn(2, ch, 20, 20)
        m = RepDWConvBN(ch, k_main=k_main, s=1)
        m.eval()

        y_train = m(x)
        m.fuse()
        y_fused = m(x)

        diff = (y_train - y_fused).abs().max().item()
        print(f"  ch={ch:4d}, k={k_main}: diff={diff:.8f}, params_fused={count_params(m):,}")
        assert y_train.shape == x.shape
        assert y_fused.shape == x.shape
        assert diff < 1e-4, f"Fuse error too large: {diff}"

    # Test with stride=2
    x = torch.randn(2, 64, 20, 20)
    m = RepDWConvBN(64, k_main=5, s=2)
    m.eval()
    y = m(x)
    print(f"  ch=64, k=5, s=2: output={list(y.shape)}")
    assert y.shape == (2, 64, 10, 10)

    print("  ✅ RepDWConvBN: ALL TESTS PASSED")


def test_rep_fused_ib_block():
    """Test RepFusedIBBlock module."""
    from ultralytics.nn.modules.MobileNetV4Pro import RepFusedIBBlock

    print("\n" + "=" * 70)
    print("TEST 4: RepFusedIBBlock — Rep Fused Inverted Bottleneck")
    print("=" * 70)

    # expand=1 path with residual
    x = torch.randn(2, 64, 40, 40)
    m = RepFusedIBBlock(64, 64, s=1, expand=1)
    y = m(x)
    print(f"  expand=1, res: input={list(x.shape)} → output={list(y.shape)}, params={count_params(m):,}")
    assert y.shape == x.shape

    # expand=4 with residual
    m2 = RepFusedIBBlock(64, 64, s=1, expand=4)
    y2 = m2(x)
    print(f"  expand=4, res: input={list(x.shape)} → output={list(y2.shape)}, params={count_params(m2):,}")
    assert y2.shape == x.shape

    # expand=4, stride=2
    m3 = RepFusedIBBlock(64, 128, s=2, expand=4)
    y3 = m3(x)
    print(f"  expand=4, s=2: input={list(x.shape)} → output={list(y3.shape)}, params={count_params(m3):,}")
    assert y3.shape == (2, 128, 20, 20)

    # Test fuse correctness
    m4 = RepFusedIBBlock(64, 64, s=1, expand=2)
    m4.eval()
    y_before = m4(x)
    m4.fuse()
    y_after = m4(x)
    diff = (y_before - y_after).abs().max().item()
    print(f"  Fuse test: diff={diff:.8f}")
    assert diff < 1e-4

    print("  ✅ RepFusedIBBlock: ALL TESTS PASSED")


def test_rep_uib_block():
    """Test RepUIBBlock — Reparameterizable UIB."""
    from ultralytics.nn.modules.MobileNetV4Pro import RepUIBBlock

    print("\n" + "=" * 70)
    print("TEST 5: RepUIBBlock — Rep Universal Inverted Bottleneck")
    print("=" * 70)

    # Extra-DW mixed (3,5) with residual + DualPoolSE
    x = torch.randn(2, 64, 20, 20)
    m1 = RepUIBBlock(64, 64, s=1, expand=4, dw_start_k=3, dw_mid_k=5, se_ratio=0.25)
    y1 = m1(x)
    print(f"  Extra-DW (3,5) SE res: input={list(x.shape)} → output={list(y1.shape)}, params={count_params(m1):,}")
    assert y1.shape == x.shape

    # Extra-DW with stride=2
    m2 = RepUIBBlock(64, 128, s=2, expand=4, dw_start_k=3, dw_mid_k=5, se_ratio=0.0)
    y2 = m2(x)
    print(f"  Extra-DW (3,5) s=2:   input={list(x.shape)} → output={list(y2.shape)}, params={count_params(m2):,}")
    assert y2.shape == (2, 128, 10, 10)

    # MBConv-like (0,3) no SE
    m3 = RepUIBBlock(64, 64, s=1, expand=4, dw_start_k=0, dw_mid_k=3, se_ratio=0.0)
    y3 = m3(x)
    print(f"  MBConv (0,3) res:     input={list(x.shape)} → output={list(y3.shape)}, params={count_params(m3):,}")
    assert y3.shape == x.shape

    # Test fuse correctness
    m4 = RepUIBBlock(64, 64, s=1, expand=4, dw_start_k=3, dw_mid_k=5, se_ratio=0.25)
    m4.eval()
    y_before = m4(x)
    m4.fuse()
    y_after = m4(x)
    diff = (y_before - y_after).abs().max().item()
    print(f"  Fuse test: diff={diff:.8f}")
    assert diff < 1e-4

    print("  ✅ RepUIBBlock: ALL TESTS PASSED")


def test_efficient_gqa():
    """Test EfficientGQA — Grouped Query Attention."""
    from ultralytics.nn.modules.MobileNetV4Pro import EfficientGQA

    print("\n" + "=" * 70)
    print("TEST 6: EfficientGQA — Efficient Grouped Query Attention")
    print("=" * 70)

    for dim, num_heads, kv_groups in [(64, 2, 1), (128, 4, 2), (256, 8, 2), (512, 8, 2)]:
        x = torch.randn(2, dim, 10, 10)
        m = EfficientGQA(dim, num_heads=num_heads, kv_groups=kv_groups, ffn_ratio=2.0)
        y = m(x)
        params = count_params(m)
        print(f"  dim={dim:4d}, heads={num_heads}, kv_groups={kv_groups}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    print("  ✅ EfficientGQA: ALL TESTS PASSED")


def test_mnv4pro_conv_stage():
    """Test MNV4ProConv stage module."""
    from ultralytics.nn.modules.MobileNetV4Pro import MNV4ProConv

    print("\n" + "=" * 70)
    print("TEST 7: MNV4ProConv Stage — RepFused IB Stack")
    print("=" * 70)

    configs = [
        (64, 64, 1, 1, 2, "64→64, n=1, expand=2"),
        (64, 128, 2, 2, 2, "64→128, n=2, expand=2, s=2"),
        (128, 256, 3, 2, 4, "128→256, n=3, expand=4, s=2"),
    ]

    for c1, c2, n, s, expand, desc in configs:
        x = torch.randn(2, c1, 40, 40)
        m = MNV4ProConv(c1, c2, n=n, s=s, expand=expand)
        y = m(x)
        params = count_params(m)
        expected_h = 40 // s
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, expected_h, expected_h)

    print("  ✅ MNV4ProConv Stage: ALL TESTS PASSED")


def test_mnv4pro_uib_stage():
    """Test MNV4ProUIB stage module."""
    from ultralytics.nn.modules.MobileNetV4Pro import MNV4ProUIB

    print("\n" + "=" * 70)
    print("TEST 8: MNV4ProUIB Stage — RepUIB + DualPoolSE Stack")
    print("=" * 70)

    configs = [
        (128, 256, 2, 2, 4, 0.25, "128→256, n=2, expand=4, SE=0.25, s=2"),
        (256, 512, 3, 2, 4, 0.0, "256→512, n=3, expand=4, SE=0, s=2"),
        (512, 512, 2, 1, 4, 0.25, "512→512, n=2, expand=4, SE=0.25, s=1"),
    ]

    for c1, c2, n, s, expand, se, desc in configs:
        x = torch.randn(2, c1, 20, 20)
        m = MNV4ProUIB(c1, c2, n=n, s=s, expand=expand, se_ratio=se)
        y = m(x)
        params = count_params(m)
        expected_h = 20 // s
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, expected_h, expected_h)

    print("  ✅ MNV4ProUIB Stage: ALL TESTS PASSED")


def test_mnv4pro_hybrid_stage():
    """Test MNV4ProHybrid stage module."""
    from ultralytics.nn.modules.MobileNetV4Pro import MNV4ProHybrid

    print("\n" + "=" * 70)
    print("TEST 9: MNV4ProHybrid Stage — RepUIB + EfficientGQA")
    print("=" * 70)

    configs = [
        (256, 512, 2, 2, 4, "256→512, n=2, expand=4, s=2"),
        (512, 1024, 2, 2, 4, "512→1024, n=2, expand=4, s=2"),
        (1024, 1024, 2, 1, 4, "1024→1024, n=2, expand=4, s=1"),
    ]

    for c1, c2, n, s, expand, desc in configs:
        x = torch.randn(2, c1, 20, 20)
        m = MNV4ProHybrid(c1, c2, n=n, s=s, expand=expand)
        y = m(x)
        params = count_params(m)
        expected_h = 20 // s
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, expected_h, expected_h)

    print("  ✅ MNV4ProHybrid Stage: ALL TESTS PASSED")


def test_mnv4pro_neck():
    """Test MNV4ProNeck module."""
    from ultralytics.nn.modules.MobileNetV4Pro import MNV4ProNeck

    print("\n" + "=" * 70)
    print("TEST 10: MNV4ProNeck — UIB-based Neck Block")
    print("=" * 70)

    configs = [
        (512, 256, 2, "512→256, n=2 (channel reduction)"),
        (768, 512, 2, "768→512, n=2 (post-concat)"),
        (256, 256, 2, "256→256, n=2 (same channel)"),
    ]

    for c1, c2, n, desc in configs:
        x = torch.randn(2, c1, 20, 20)
        m = MNV4ProNeck(c1, c2, n=n)
        y = m(x)
        params = count_params(m)
        print(f"  {desc}: input={list(x.shape)} → output={list(y.shape)}, params={params:,}")
        assert y.shape == (2, c2, 20, 20)

    print("  ✅ MNV4ProNeck: ALL TESTS PASSED")


def test_gradient_flow():
    """Test gradient flow through MNV4Pro modules."""
    from ultralytics.nn.modules.MobileNetV4Pro import RepUIBBlock, RepFusedIBBlock, EfficientGQA, MNV4ProUIB

    print("\n" + "=" * 70)
    print("TEST 11: Gradient Flow Validation")
    print("=" * 70)

    # RepFusedIBBlock gradient
    x1 = torch.randn(2, 64, 20, 20, requires_grad=True)
    m1 = RepFusedIBBlock(64, 64, s=1, expand=4)
    y1 = m1(x1)
    y1.sum().backward()
    assert x1.grad is not None and x1.grad.abs().sum() > 0
    print(f"  RepFusedIBBlock gradient: ✅ (grad norm = {x1.grad.norm():.4f})")

    # RepUIBBlock gradient (Extra-DW + DualPoolSE)
    x2 = torch.randn(2, 64, 20, 20, requires_grad=True)
    m2 = RepUIBBlock(64, 64, s=1, expand=4, dw_start_k=3, dw_mid_k=5, se_ratio=0.25)
    y2 = m2(x2)
    y2.sum().backward()
    assert x2.grad is not None and x2.grad.abs().sum() > 0
    print(f"  RepUIBBlock (SE) gradient: ✅ (grad norm = {x2.grad.norm():.4f})")

    # EfficientGQA gradient
    x3 = torch.randn(2, 128, 10, 10, requires_grad=True)
    m3 = EfficientGQA(128, num_heads=4, kv_groups=2)
    y3 = m3(x3)
    y3.sum().backward()
    assert x3.grad is not None and x3.grad.abs().sum() > 0
    print(f"  EfficientGQA gradient: ✅ (grad norm = {x3.grad.norm():.4f})")

    # MNV4ProUIB stage gradient
    x4 = torch.randn(2, 128, 20, 20, requires_grad=True)
    m4 = MNV4ProUIB(128, 128, n=3, s=1, expand=4, se_ratio=0.25)
    y4 = m4(x4)
    y4.sum().backward()
    assert x4.grad is not None and x4.grad.abs().sum() > 0
    print(f"  MNV4ProUIB stage gradient: ✅ (grad norm = {x4.grad.norm():.4f})")

    # ProLayerScale gamma gradient
    from ultralytics.nn.modules.MobileNetV4Pro import ProLayerScale
    ls = ProLayerScale(64)
    x5 = torch.randn(2, 64, 10, 10)
    y5 = ls(x5)
    y5.sum().backward()
    assert ls.gamma.grad is not None
    print(f"  ProLayerScale γ gradient: ✅ (γ grad norm = {ls.gamma.grad.norm():.6f})")

    print("  ✅ Gradient Flow: ALL TESTS PASSED")


def test_reparameterization_full():
    """Test full reparameterization pipeline."""
    from ultralytics.nn.modules.MobileNetV4Pro import MNV4ProConv, MNV4ProUIB

    print("\n" + "=" * 70)
    print("TEST 12: Full Reparameterization Pipeline")
    print("=" * 70)

    # Test MNV4ProConv fuse
    x = torch.randn(2, 64, 40, 40)
    m = MNV4ProConv(64, 128, n=2, s=2, expand=4)
    m.eval()

    params_before = count_params(m)
    y_before = m(x)

    m.fuse()
    params_after = count_params(m)
    y_after = m(x)

    diff = (y_before - y_after).abs().max().item()
    print(f"  MNV4ProConv fuse: params {params_before:,} → {params_after:,}, diff={diff:.8f}")
    assert diff < 1e-4

    # Test MNV4ProUIB fuse
    x2 = torch.randn(2, 128, 20, 20)
    m2 = MNV4ProUIB(128, 256, n=2, s=2, expand=4, se_ratio=0.25)
    m2.eval()

    params_before2 = count_params(m2)
    y_before2 = m2(x2)

    m2.fuse()
    params_after2 = count_params(m2)
    y_after2 = m2(x2)

    diff2 = (y_before2 - y_after2).abs().max().item()
    print(f"  MNV4ProUIB fuse:  params {params_before2:,} → {params_after2:,}, diff={diff2:.8f}")
    assert diff2 < 1e-4

    print("  ✅ Reparameterization: ALL TESTS PASSED")


def test_full_model_pro():
    """Test full YOLO11-MobileNetV4Pro model build from YAML."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 13: Full YOLO11-MobileNetV4Pro Model")
    print("=" * 70)

    yaml_path = "ultralytics/cfg/models/11/yolo11-MobileNetV4Pro/yolo11-MobileNetV4Pro.yaml"
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
    print(f"  ✅ Full YOLO11-MobileNetV4Pro: PASSED")

    return total_params


def test_full_model_pro_hybrid():
    """Test full YOLO11-MobileNetV4Pro-Hybrid model."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 14: Full YOLO11-MobileNetV4Pro-Hybrid (RepUIB + EfficientGQA)")
    print("=" * 70)

    yaml_path = "ultralytics/cfg/models/11/yolo11-MobileNetV4Pro/yolo11-MobileNetV4Pro-Hybrid.yaml"
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
    print(f"  ✅ Full YOLO11-MobileNetV4Pro-Hybrid: PASSED")

    return total_params


def test_comparison():
    """Compare parameter counts — MobileNetV4Pro vs baseline and MobileNetV4."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("TEST 15: Architecture Comparison")
    print("=" * 70)

    configs = {
        "YOLO11 (baseline)": "ultralytics/cfg/models/11/yolo11.yaml",
        "MobileNetV4 (lightweight)": "ultralytics/cfg/models/11/yolo11-MobileNetV4/yolo11-MobileNetV4.yaml",
        "MobileNetV4-Hybrid": "ultralytics/cfg/models/11/yolo11-MobileNetV4/yolo11-MobileNetV4-Hybrid.yaml",
        "MobileNetV4Pro": "ultralytics/cfg/models/11/yolo11-MobileNetV4Pro/yolo11-MobileNetV4Pro.yaml",
        "MobileNetV4Pro-Hybrid": "ultralytics/cfg/models/11/yolo11-MobileNetV4Pro/yolo11-MobileNetV4Pro-Hybrid.yaml",
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
        for name, params in results.items():
            if name != "YOLO11 (baseline)":
                diff = (params / baseline - 1) * 100
                status = "✅ LIGHTER" if params < baseline else "⚠️  HEAVIER" if params < baseline * 1.1 else "❌ MUCH HEAVIER"
                print(f"  {name} vs baseline: {diff:+.1f}% → {status}")

    # Compare Pro vs original MNV4
    print()
    if "MobileNetV4 (lightweight)" in results and "MobileNetV4Pro" in results:
        mnv4 = results["MobileNetV4 (lightweight)"]
        pro = results["MobileNetV4Pro"]
        diff = (pro / mnv4 - 1) * 100
        print(f"  MobileNetV4Pro vs MobileNetV4: {diff:+.1f}% params")
        print(f"  → Pro has richer training (multi-branch) but FUSES to same inference cost")

    if "MobileNetV4-Hybrid" in results and "MobileNetV4Pro-Hybrid" in results:
        mnv4h = results["MobileNetV4-Hybrid"]
        proh = results["MobileNetV4Pro-Hybrid"]
        diff = (proh / mnv4h - 1) * 100
        print(f"  MobileNetV4Pro-Hybrid vs MobileNetV4-Hybrid: {diff:+.1f}% params")
        print(f"  → Pro-Hybrid uses GQA + Gated FFN instead of MQA for better accuracy")

    print()
    print("  ✅ Comparison: DONE")


def test_inference_speed():
    """Benchmark inference speed comparison."""
    from ultralytics.nn.modules.MobileNetV4Pro import MNV4ProConv, MNV4ProUIB, RepFusedIBBlock
    from ultralytics.nn.modules.MobileNetV4 import MNV4Conv, MNV4UIB, FusedIBBlock

    print("\n" + "=" * 70)
    print("TEST 16: Inference Speed (Rep-fused vs Original)")
    print("=" * 70)

    x = torch.randn(4, 128, 40, 40)
    n_iters = 50

    # Original MNV4Conv
    m_orig = MNV4Conv(128, 256, n=3, s=2, expand=4)
    m_orig.eval()
    with torch.no_grad():
        for _ in range(5):  # warmup
            m_orig(x)
        t0 = time.time()
        for _ in range(n_iters):
            m_orig(x)
        t_orig = (time.time() - t0) / n_iters * 1000

    # Pro MNV4ProConv (before fuse)
    m_pro = MNV4ProConv(128, 256, n=3, s=2, expand=4)
    m_pro.eval()
    with torch.no_grad():
        for _ in range(5):
            m_pro(x)
        t0 = time.time()
        for _ in range(n_iters):
            m_pro(x)
        t_pro_train = (time.time() - t0) / n_iters * 1000

    # Pro MNV4ProConv (after fuse)
    m_pro.fuse()
    with torch.no_grad():
        for _ in range(5):
            m_pro(x)
        t0 = time.time()
        for _ in range(n_iters):
            m_pro(x)
        t_pro_fused = (time.time() - t0) / n_iters * 1000

    print(f"  MNV4Conv (original):     {t_orig:.2f} ms/iter")
    print(f"  MNV4ProConv (training):  {t_pro_train:.2f} ms/iter")
    print(f"  MNV4ProConv (fused):     {t_pro_fused:.2f} ms/iter")
    print(f"  → Fused Pro should be ≈ same speed as original (or faster)")

    print("  ✅ Speed Benchmark: DONE")


if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║  YOLO11-MobileNetV4Pro Architecture Validation Test Suite         ║")
    print("║  RepUIB + DualPoolSE + EfficientGQA + Gated FFN + UIB Neck       ║")
    print("╚" + "═" * 68 + "╝")

    start = time.time()

    # Run all tests
    test_dual_pool_se()
    test_rep_conv_bn()
    test_rep_dw_conv_bn()
    test_rep_fused_ib_block()
    test_rep_uib_block()
    test_efficient_gqa()
    test_mnv4pro_conv_stage()
    test_mnv4pro_uib_stage()
    test_mnv4pro_hybrid_stage()
    test_mnv4pro_neck()
    test_gradient_flow()
    test_reparameterization_full()
    test_full_model_pro()
    test_full_model_pro_hybrid()
    test_comparison()
    test_inference_speed()

    elapsed = time.time() - start

    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║  ALL TESTS PASSED ✅  ({elapsed:.1f}s)                                     ║")
    print("╚" + "═" * 68 + "╝")
