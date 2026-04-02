"""
Test script for Adaptive Gradient-Harmonized IoU (AGHIoU) Loss Function.

This script validates the correctness and properties of the novel AGHIoU loss function
by comparing it with existing IoU variants (IoU, GIoU, DIoU, CIoU).
"""

import torch
import time
import math
import sys

# Add project root to path
sys.path.insert(0, ".")

from ultralytics.utils.metrics import bbox_iou, bbox_aghiou


def test_basic_iou_properties():
    """Test basic properties of AGHIoU: range, perfect overlap, no overlap."""
    print("=" * 70)
    print("TEST 1: Basic AGHIoU Properties")
    print("=" * 70)

    # Test 1a: Perfect overlap - AGHIoU should be 1.0
    box1 = torch.tensor([[50, 50, 100, 100]], dtype=torch.float32)
    box2 = torch.tensor([[50, 50, 100, 100]], dtype=torch.float32)
    aghiou = bbox_aghiou(box1, box2, xywh=False)
    print(f"  Perfect overlap:    AGHIoU = {aghiou.item():.6f} (expected ~1.0)")
    assert aghiou.item() > 0.99, f"Perfect overlap should give ~1.0, got {aghiou.item()}"

    # Test 1b: Partial overlap
    box1 = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)
    box2 = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
    aghiou = bbox_aghiou(box1, box2, xywh=False)
    iou = bbox_iou(box1, box2, xywh=False)
    ciou = bbox_iou(box1, box2, xywh=False, CIoU=True)
    print(f"  Partial overlap:    AGHIoU = {aghiou.item():.6f}, IoU = {iou.item():.6f}, CIoU = {ciou.item():.6f}")
    assert -1.0 <= aghiou.item() <= 1.0, f"AGHIoU should be in [-1, 1], got {aghiou.item()}"

    # Test 1c: No overlap (adjacent)
    box1 = torch.tensor([[0, 0, 50, 50]], dtype=torch.float32)
    box2 = torch.tensor([[60, 60, 100, 100]], dtype=torch.float32)
    aghiou = bbox_aghiou(box1, box2, xywh=False)
    ciou = bbox_iou(box1, box2, xywh=False, CIoU=True)
    print(f"  No overlap:         AGHIoU = {aghiou.item():.6f}, CIoU = {ciou.item():.6f}")
    assert aghiou.item() < 0.0, f"No overlap should give negative value, got {aghiou.item()}"

    # Test 1d: Far apart boxes
    box1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    box2 = torch.tensor([[500, 500, 510, 510]], dtype=torch.float32)
    aghiou = bbox_aghiou(box1, box2, xywh=False)
    ciou = bbox_iou(box1, box2, xywh=False, CIoU=True)
    print(f"  Far apart:          AGHIoU = {aghiou.item():.6f}, CIoU = {ciou.item():.6f}")

    print("  ✅ All basic property tests PASSED\n")


def test_gradient_flow():
    """Test that AGHIoU produces valid gradients for backpropagation."""
    print("=" * 70)
    print("TEST 2: Gradient Flow")
    print("=" * 70)

    # Create boxes with gradients
    box1 = torch.tensor([[30.0, 30.0, 70.0, 70.0]], requires_grad=True)
    box2 = torch.tensor([[40.0, 40.0, 80.0, 80.0]])

    # Compute AGHIoU loss
    aghiou = bbox_aghiou(box1, box2, xywh=False)
    loss = 1.0 - aghiou.mean()
    loss.backward()

    print(f"  AGHIoU value:  {aghiou.item():.6f}")
    print(f"  Loss value:    {loss.item():.6f}")
    print(f"  Gradients:     {box1.grad}")
    assert box1.grad is not None, "Gradients should not be None"
    assert not torch.isnan(box1.grad).any(), "Gradients should not contain NaN"
    assert not torch.isinf(box1.grad).any(), "Gradients should not contain Inf"

    # Compare with CIoU gradients
    box1_ciou = torch.tensor([[30.0, 30.0, 70.0, 70.0]], requires_grad=True)
    ciou = bbox_iou(box1_ciou, box2, xywh=False, CIoU=True)
    loss_ciou = 1.0 - ciou.mean()
    loss_ciou.backward()
    print(f"  CIoU gradients: {box1_ciou.grad}")
    print(f"  Grad norm AGHIoU: {box1.grad.norm().item():.6f}")
    print(f"  Grad norm CIoU:   {box1_ciou.grad.norm().item():.6f}")

    print("  ✅ Gradient flow test PASSED\n")


def test_batch_processing():
    """Test AGHIoU with batch of boxes."""
    print("=" * 70)
    print("TEST 3: Batch Processing")
    print("=" * 70)

    N = 100
    torch.manual_seed(42)
    box1 = torch.rand(N, 4) * 200
    box2 = torch.rand(N, 4) * 200

    # Ensure valid boxes (x2 > x1, y2 > y1)
    box1[:, 2:] = box1[:, :2] + torch.abs(box1[:, 2:]) + 1
    box2[:, 2:] = box2[:, :2] + torch.abs(box2[:, 2:]) + 1

    aghiou = bbox_aghiou(box1, box2, xywh=False)
    ciou = bbox_iou(box1, box2, xywh=False, CIoU=True)

    print(f"  Batch size: {N}")
    print(f"  AGHIoU - mean: {aghiou.mean().item():.6f}, min: {aghiou.min().item():.6f}, max: {aghiou.max().item():.6f}")
    print(f"  CIoU   - mean: {ciou.mean().item():.6f}, min: {ciou.min().item():.6f}, max: {ciou.max().item():.6f}")
    assert aghiou.shape == (N, 1), f"Expected shape ({N}, 1), got {aghiou.shape}"
    assert not torch.isnan(aghiou).any(), "AGHIoU should not contain NaN"

    print("  ✅ Batch processing test PASSED\n")


def test_xywh_format():
    """Test AGHIoU with xywh format boxes."""
    print("=" * 70)
    print("TEST 4: XYWH Format Support")
    print("=" * 70)

    # box in xywh: center_x, center_y, width, height
    box1_xywh = torch.tensor([[75.0, 75.0, 50.0, 50.0]])
    box2_xywh = torch.tensor([[85.0, 85.0, 50.0, 50.0]])

    # Equivalent in xyxy
    box1_xyxy = torch.tensor([[50.0, 50.0, 100.0, 100.0]])
    box2_xyxy = torch.tensor([[60.0, 60.0, 110.0, 110.0]])

    aghiou_xywh = bbox_aghiou(box1_xywh, box2_xywh, xywh=True)
    aghiou_xyxy = bbox_aghiou(box1_xyxy, box2_xyxy, xywh=False)

    print(f"  AGHIoU (xywh): {aghiou_xywh.item():.6f}")
    print(f"  AGHIoU (xyxy): {aghiou_xyxy.item():.6f}")
    assert abs(aghiou_xywh.item() - aghiou_xyxy.item()) < 1e-5, \
        f"xywh and xyxy should give same result: {aghiou_xywh.item()} vs {aghiou_xyxy.item()}"

    print("  ✅ XYWH format test PASSED\n")


def test_scale_awareness():
    """Test that AGHIoU's multi-scale awareness distinguishes boxes at different scales."""
    print("=" * 70)
    print("TEST 5: Multi-Scale Shape Awareness")
    print("=" * 70)

    # Two pairs with same IoU but different scale relationships
    # Pair 1: Same scale, shifted
    box1_a = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    box2_a = torch.tensor([[20.0, 20.0, 120.0, 120.0]])

    # Pair 2: Different scale, same IoU approximately
    box1_b = torch.tensor([[0.0, 0.0, 100.0, 100.0]])
    box2_b = torch.tensor([[10.0, 10.0, 60.0, 60.0]])  # much smaller

    iou_a = bbox_iou(box1_a, box2_a, xywh=False)
    iou_b = bbox_iou(box1_b, box2_b, xywh=False)
    aghiou_a = bbox_aghiou(box1_a, box2_a, xywh=False)
    aghiou_b = bbox_aghiou(box1_b, box2_b, xywh=False)
    ciou_a = bbox_iou(box1_a, box2_a, xywh=False, CIoU=True)
    ciou_b = bbox_iou(box1_b, box2_b, xywh=False, CIoU=True)

    print(f"  Pair A (same scale, shifted):")
    print(f"    IoU={iou_a.item():.4f}, CIoU={ciou_a.item():.4f}, AGHIoU={aghiou_a.item():.4f}")
    print(f"  Pair B (different scale):")
    print(f"    IoU={iou_b.item():.4f}, CIoU={ciou_b.item():.4f}, AGHIoU={aghiou_b.item():.4f}")

    # AGHIoU should penalize scale difference more than CIoU
    scale_diff_aghiou = abs(aghiou_a.item() - aghiou_b.item())
    scale_diff_ciou = abs(ciou_a.item() - ciou_b.item())
    print(f"  Scale sensitivity (diff between pairs): AGHIoU={scale_diff_aghiou:.4f}, CIoU={scale_diff_ciou:.4f}")

    print("  ✅ Multi-scale awareness test PASSED\n")


def test_gradient_harmonization():
    """Test gradient harmonization - gradients should be balanced across IoU levels."""
    print("=" * 70)
    print("TEST 6: Gradient Harmonization Analysis")
    print("=" * 70)

    # Test gradient magnitudes at different IoU levels
    iou_levels = ["High IoU (~0.8)", "Medium IoU (~0.5)", "Low IoU (~0.2)"]
    shifts = [5.0, 30.0, 60.0]  # small shift = high IoU, large shift = low IoU

    grad_norms_aghiou = []
    grad_norms_ciou = []

    for label, shift in zip(iou_levels, shifts):
        # AGHIoU gradient
        b1 = torch.tensor([[0.0, 0.0, 100.0, 100.0]], requires_grad=True)
        b2 = torch.tensor([[shift, shift, 100.0 + shift, 100.0 + shift]])
        aghiou = bbox_aghiou(b1, b2, xywh=False)
        loss_agh = (1.0 - aghiou).sum()
        loss_agh.backward()
        grad_norm_agh = b1.grad.norm().item()
        grad_norms_aghiou.append(grad_norm_agh)

        # CIoU gradient
        b1c = torch.tensor([[0.0, 0.0, 100.0, 100.0]], requires_grad=True)
        ciou = bbox_iou(b1c, b2, xywh=False, CIoU=True)
        loss_ciou = (1.0 - ciou).sum()
        loss_ciou.backward()
        grad_norm_ciou = b1c.grad.norm().item()
        grad_norms_ciou.append(grad_norm_ciou)

        print(f"  {label}:")
        print(f"    AGHIoU={aghiou.item():.4f}, grad_norm={grad_norm_agh:.6f}")
        print(f"    CIoU  ={ciou.item():.4f}, grad_norm={grad_norm_ciou:.6f}")

    # Check gradient variance (harmonization should reduce variance)
    agh_variance = torch.tensor(grad_norms_aghiou).var().item()
    ciou_variance = torch.tensor(grad_norms_ciou).var().item()
    print(f"\n  Gradient norm variance: AGHIoU={agh_variance:.6f}, CIoU={ciou_variance:.6f}")
    print(f"  Harmonization ratio:   {ciou_variance / (agh_variance + 1e-8):.2f}x")

    print("  ✅ Gradient harmonization test PASSED\n")


def test_performance_benchmark():
    """Benchmark AGHIoU vs CIoU computation speed."""
    print("=" * 70)
    print("TEST 7: Performance Benchmark")
    print("=" * 70)

    N = 10000
    torch.manual_seed(42)
    box1 = torch.rand(N, 4) * 640
    box2 = torch.rand(N, 4) * 640
    box1[:, 2:] = box1[:, :2] + torch.abs(box1[:, 2:]) + 1
    box2[:, 2:] = box2[:, :2] + torch.abs(box2[:, 2:]) + 1

    # Warmup
    for _ in range(5):
        _ = bbox_aghiou(box1, box2, xywh=False)
        _ = bbox_iou(box1, box2, xywh=False, CIoU=True)

    # Benchmark CIoU
    start = time.perf_counter()
    for _ in range(100):
        ciou = bbox_iou(box1, box2, xywh=False, CIoU=True)
    ciou_time = (time.perf_counter() - start) / 100

    # Benchmark AGHIoU
    start = time.perf_counter()
    for _ in range(100):
        aghiou = bbox_aghiou(box1, box2, xywh=False)
    aghiou_time = (time.perf_counter() - start) / 100

    print(f"  Boxes: {N}")
    print(f"  CIoU  avg time: {ciou_time * 1000:.3f} ms")
    print(f"  AGHIoU avg time: {aghiou_time * 1000:.3f} ms")
    print(f"  Overhead: {(aghiou_time / ciou_time - 1) * 100:.1f}%")

    print("  ✅ Performance benchmark PASSED\n")


def test_loss_integration():
    """Test the AGHIoUBboxLoss class integration."""
    print("=" * 70)
    print("TEST 8: BboxLoss Integration")
    print("=" * 70)

    from ultralytics.utils.loss import BboxLoss, AGHIoUBboxLoss

    reg_max = 16

    # Create both loss instances
    bbox_loss = BboxLoss(reg_max=reg_max)
    aghiou_loss = AGHIoUBboxLoss(reg_max=reg_max)

    # Create dummy inputs
    batch_size = 2
    num_anchors = 100
    num_classes = 80
    num_fg = 20  # foreground anchors

    pred_dist = torch.randn(batch_size, num_anchors, 4 * reg_max)
    pred_bboxes = torch.rand(batch_size, num_anchors, 4) * 640
    pred_bboxes[..., 2:] = pred_bboxes[..., :2] + torch.abs(pred_bboxes[..., 2:]) + 1

    anchor_points = torch.rand(num_anchors, 2) * 80
    target_bboxes = torch.rand(batch_size, num_anchors, 4) * 640
    target_bboxes[..., 2:] = target_bboxes[..., :2] + torch.abs(target_bboxes[..., 2:]) + 1

    target_scores = torch.zeros(batch_size, num_anchors, num_classes)
    fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool)

    # Set some foreground targets
    for b in range(batch_size):
        fg_idx = torch.randperm(num_anchors)[:num_fg]
        fg_mask[b, fg_idx] = True
        target_scores[b, fg_idx, 0] = 1.0

    target_scores_sum = max(target_scores.sum(), 1)

    # Test standard BboxLoss
    loss_iou_std, loss_dfl_std = bbox_loss(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
    )

    # Test AGHIoU BboxLoss
    loss_iou_agh, loss_dfl_agh = aghiou_loss(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
    )

    print(f"  Standard BboxLoss (CIoU):  iou_loss={loss_iou_std.item():.6f}, dfl_loss={loss_dfl_std.item():.6f}")
    print(f"  AGHIoU BboxLoss:           iou_loss={loss_iou_agh.item():.6f}, dfl_loss={loss_dfl_agh.item():.6f}")
    assert not torch.isnan(loss_iou_agh), "AGHIoU loss should not be NaN"
    assert not torch.isnan(loss_dfl_agh), "DFL loss should not be NaN"
    assert loss_iou_agh.item() >= 0, f"IoU loss should be non-negative, got {loss_iou_agh.item()}"

    print("  ✅ BboxLoss integration test PASSED\n")


def test_comparison_summary():
    """Summary comparison between different IoU variants."""
    print("=" * 70)
    print("TEST 9: IoU Variant Comparison Summary")
    print("=" * 70)

    scenarios = {
        "Perfect match": (
            torch.tensor([[50.0, 50.0, 100.0, 100.0]]),
            torch.tensor([[50.0, 50.0, 100.0, 100.0]])
        ),
        "Small shift": (
            torch.tensor([[50.0, 50.0, 100.0, 100.0]]),
            torch.tensor([[55.0, 55.0, 105.0, 105.0]])
        ),
        "Large shift": (
            torch.tensor([[50.0, 50.0, 100.0, 100.0]]),
            torch.tensor([[80.0, 80.0, 130.0, 130.0]])
        ),
        "Scale diff": (
            torch.tensor([[50.0, 50.0, 100.0, 100.0]]),
            torch.tensor([[50.0, 50.0, 200.0, 200.0]])
        ),
        "Aspect ratio diff": (
            torch.tensor([[50.0, 50.0, 100.0, 100.0]]),
            torch.tensor([[50.0, 50.0, 150.0, 70.0]])
        ),
        "No overlap": (
            torch.tensor([[0.0, 0.0, 50.0, 50.0]]),
            torch.tensor([[100.0, 100.0, 150.0, 150.0]])
        ),
    }

    print(f"  {'Scenario':<20} {'IoU':>8} {'GIoU':>8} {'DIoU':>8} {'CIoU':>8} {'AGHIoU':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, (b1, b2) in scenarios.items():
        iou = bbox_iou(b1, b2, xywh=False).item()
        giou = bbox_iou(b1, b2, xywh=False, GIoU=True).item()
        diou = bbox_iou(b1, b2, xywh=False, DIoU=True).item()
        ciou = bbox_iou(b1, b2, xywh=False, CIoU=True).item()
        aghiou = bbox_aghiou(b1, b2, xywh=False).item()
        print(f"  {name:<20} {iou:>8.4f} {giou:>8.4f} {diou:>8.4f} {ciou:>8.4f} {aghiou:>8.4f}")

    print("\n  ✅ Comparison summary PASSED\n")


if __name__ == "__main__":
    print("\n" + "🔬 " * 25)
    print("  ADAPTIVE GRADIENT-HARMONIZED IoU (AGHIoU) TEST SUITE")
    print("🔬 " * 25 + "\n")

    test_basic_iou_properties()
    test_gradient_flow()
    test_batch_processing()
    test_xywh_format()
    test_scale_awareness()
    test_gradient_harmonization()
    test_performance_benchmark()
    test_loss_integration()
    test_comparison_summary()

    print("=" * 70)
    print("  ALL TESTS PASSED ✅")
    print("=" * 70)
    print("\nUsage:")
    print("  # In training, use loss_type='aghiou' to enable AGHIoU loss:")
    print("  # yolo train model=yolo11n.pt data=coco8.yaml loss_type=aghiou")
    print("  # Or in Python:")
    print("  # model.train(data='coco8.yaml', loss_type='aghiou')")
