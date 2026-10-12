"""
Test script for YOLO11 + ViT-Swin Hybrid Backbone
===================================================
Kiểm tra kiến trúc lai kết hợp ViT và Swin Transformer:
- Swin @ P3-P4 early (efficient window attention cho resolution cao)
- ViT @ P4 late-P5 (global attention cho semantic features)
- Neck/Head giữ nguyên YOLO11 chuẩn
"""

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import SwinStage, ViTStage

def test_individual_modules():
    """Test riêng từng module Transformer"""
    print("\n" + "="*70)
    print("KIỂM TRA MODULES RỜI")
    print("="*70)

    batch_size = 2
    device = 'cpu'

    # Test SwinStage
    print("\n[1] Testing SwinStage (P3 stage)...")
    print("-" * 50)
    swin_p3 = SwinStage(c1=256, c2=256, n=2, s=2, window_size=7).to(device)
    x_p3 = torch.randn(batch_size, 256, 80, 80).to(device)

    with torch.no_grad():
        out_swin = swin_p3(x_p3)

    print(f"  Input:  {tuple(x_p3.shape)} (256 channels @ 80×80)")
    print(f"  Output: {tuple(out_swin.shape)} (stride=2 downsampling)")
    print(f"  Params: {sum(p.numel() for p in swin_p3.parameters()):,}")
    print("  ✓ SwinStage hoạt động OK")

    # Test ViTStage
    print("\n[2] Testing ViTStage (P5 stage)...")
    print("-" * 50)
    vit_p5 = ViTStage(c1=512, c2=1024, n=2, s=2).to(device)
    x_p5 = torch.randn(batch_size, 512, 40, 40).to(device)

    with torch.no_grad():
        out_vit = vit_p5(x_p5)

    print(f"  Input:  {tuple(x_p5.shape)} (512 channels @ 40×40)")
    print(f"  Output: {tuple(out_vit.shape)} (stride=2 downsampling)")
    print(f"  Params: {sum(p.numel() for p in vit_p5.parameters()):,}")
    print("  ✓ ViTStage hoạt động OK")

    print("\n" + "="*70)


def test_full_model():
    """Test full YOLO11 ViT-Swin Hybrid model"""
    print("\n" + "="*70)
    print("KIỂM TRA FULL MODEL: YOLO11 + ViT-Swin Hybrid")
    print("="*70)

    yaml_path = 'ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-ViT-Swin-Hybrid.yaml'

    print(f"\n[1] Đang load model từ: {yaml_path}")
    print("-" * 70)

    try:
        model = YOLO(yaml_path)
        print("  ✓ Model loaded thành công!")
    except Exception as e:
        print(f"  ✗ Lỗi khi load model: {e}")
        return

    # Model info
    print("\n[2] Thông tin model:")
    print("-" * 70)
    model.info(detailed=False, verbose=True)

    # Forward pass test
    print("\n[3] Testing forward pass...")
    print("-" * 70)

    batch_size = 2
    img_size = 640
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    print(f"  Input shape: {tuple(dummy_input.shape)} (batch={batch_size}, imgsz={img_size})")

    try:
        model.model.eval()
        with torch.no_grad():
            outputs = model.model(dummy_input)

        print(f"\n  ✓ Forward pass thành công!")
        print(f"  Number of detection heads: {len(outputs)}")

        if isinstance(outputs, (list, tuple)):
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    print(f"    - Head {i}: {tuple(out.shape)}")

    except Exception as e:
        print(f"  ✗ Lỗi forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # Parameter count
    print("\n[4] Thống kê parameters:")
    print("-" * 70)

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (MB):      {total_params * 4 / 1024 / 1024:.2f}")

    print("\n" + "="*70)
    print("✓ TẤT CẢ TESTS HOÀN THÀNH THÀNH CÔNG!")
    print("="*70)


def compare_architectures():
    """So sánh parameter count giữa các variants"""
    print("\n" + "="*70)
    print("SO SÁNH CÁC TRANSFORMER VARIANTS")
    print("="*70)

    variants = {
        'YOLO11 Standard': 'ultralytics/cfg/models/11/yolo11.yaml',
        'YOLO11-Swin': 'ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-Swin.yaml',
        'YOLO11-ViT': 'ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-ViT.yaml',
        'YOLO11-MobileFormer': 'ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-MobileFormer.yaml',
        'YOLO11-ViT-Swin-Hybrid': 'ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-ViT-Swin-Hybrid.yaml',
    }

    print("\n{:<30} {:>15} {:>12}".format("Architecture", "Parameters", "Size (MB)"))
    print("-" * 70)

    for name, yaml_path in variants.items():
        try:
            model = YOLO(yaml_path)
            total_params = sum(p.numel() for p in model.model.parameters())
            size_mb = total_params * 4 / 1024 / 1024
            print("{:<30} {:>15,} {:>12.2f}".format(name, total_params, size_mb))
        except Exception as e:
            print("{:<30} {:>15} {:>12}".format(name, "ERROR", "-"))

    print("="*70)


if __name__ == '__main__':
    print("\n" + "🚀"*35)
    print("YOLO11 + ViT-Swin Hybrid Test Suite")
    print("🚀"*35)

    # Test 1: Individual modules
    test_individual_modules()

    # Test 2: Full model
    test_full_model()

    # Test 3: Compare architectures
    compare_architectures()

    print("\n" + "✅"*35)
    print("TEST SUITE HOÀN TẤT!")
    print("✅"*35 + "\n")
