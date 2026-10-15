
import os
from pathlib import Path
import torch
from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-SHSA.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-test-17.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-unetv2.yaml')
    # model = YOLO(r'ultralytics/cfg/models/v12/yolov12.yaml')
    # model = YOLO(r'ultralytics/cfg/models/v12/yolov12-test-02.yaml')
    # model = YOLO(r'ultralytics/cfg/models/v12/yolov12-bifpn-p2-ca.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-p2-bifpn-mhsa.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-test-new-01.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-cbam-v2.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-EffecientNetV2/yolo11-EfficientNetV2-CBAM.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-EffecientNetV2/yolo11-EfficientNetV2_C2k3-DCNF-v2-P2.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-EffecientNetV2/yolo11-EfficientNetV2_C2k3-DCNF-v6.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-EDGE/yolo11-EDGE.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-Phoenix/yolo11-Phoenix.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-Chimera/yolo11-Chimera.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-Nexus/yolo11-Nexus.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-Nexus/yolo11-EfficientNetV2-Nexus.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-MobileNetV4/yolo11-MobileNetV4.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-MobileNetV4/yolo11-MobileNetV4-Hybrid.yaml')
    model = YOLO(r'ultralytics/cfg/models/11/yolo11-Prism/yolo11-Prism.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-Prism/yolo11-EfficientNetV2-Prism-v2.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-Zenith/yolo11-Zenith.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-ViT-Optimized.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-MobileFormer.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-Swin.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-TransformerHybrid/yolo11-ViT.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-EffecientNetV2/yolo11-EfficientNetV2_C2k3-DCNF-v1Plus.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-DCNF-V1Plus.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-IDC.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-EfficientNetV2-CA.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-EfficientNetV2-BiFPN.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-EfficientNetV2-Bifpn-CBAM.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-test-19.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-test-26.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/test/yolo11-bifpn-mhsa.yaml')
    # model = YOLO(r'ultralytics/cfg/models/v9/yolov9t.yaml')
    # model = YOLO(r'ultralytics/cfg/models/v10/yolov10n.yaml')

    # -------------------------------------------------------------------------
    # 1. Dataset Path Resolution (Tương thích cả Kaggle và Local máy cá nhân)
    # -------------------------------------------------------------------------
    kaggle_data = Path('/kaggle/input/bccd-new/data.yaml')
    local_data = Path(__file__).resolve().parent / 'ultralytics' / 'data' / 'bccd_detection' / 'data.yaml'

    if kaggle_data.exists():
        data_yaml = str(kaggle_data)
    elif local_data.exists():
        data_yaml = str(local_data)
    else:
        data_yaml = r'ultralytics/data/bccd_detection/data.yaml'

    # -------------------------------------------------------------------------
    # 2. Hardware & Device Setup (Cố định huấn luyện trên GPU 0)
    # -------------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n[LỖI] PyTorch hiện tại không tìm thấy GPU CUDA (đang là bản torch CPU)!\n"
            "Để train trên GPU (RTX 5090 / NVIDIA GPU), vui lòng cài đặt PyTorch CUDA bằng lệnh:\n"
            "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
        )

    device = 0                         # Cố định GPU 0 (NVIDIA RTX 5090 / CUDA:0)
    workers = 4 if os.name == 'nt' else 8

    # -------------------------------------------------------------------------
    # 3. Chuẩn hóa siêu tham số huấn luyện (Standardized Paper Protocol)
    # -------------------------------------------------------------------------
    model.train(
        # --- Cấu hình dữ liệu & phần cứng ---
        data=data_yaml,
        imgsz=640,                     # Kích thước ảnh chuẩn benchmark 640x640
        batch=8,                       # Batch size chuẩn hóa (8 samples/step)
        epochs=200,                    # 200 epochs đảm bảo hội tụ đầy đủ trên tập tế bào
        device=device,                 # 0 (GPU) hoặc 'cpu'
        workers=workers,               # Số luồng DataLoader phù hợp OS
        
        # --- Optimizer & Learning Rate Schedule (Chuẩn hóa cho AdamW) ---
        optimizer='AdamW',             # Decoupled Weight Decay Optimizer
        lr0=0.001,                     # Learning rate khởi tạo chuẩn cho AdamW (1e-3)
        lrf=0.01,                      # Final LR factor (lr_min = lr0 * lrf = 1e-5)
        momentum=0.9,                  # Momentum cho optimizer
        weight_decay=0.01,             # L2 regularization chống overfitting trên tập BCCD
        warmup_epochs=3.0,             # 3 epoch warm-up tuyến tính để ổn định attention/backbone
        warmup_momentum=0.8,           # Momentum trong giai đoạn warm-up
        warmup_bias_lr=0.1,            # Bias learning rate trong warm-up
        cos_lr=True,                   # Cosine Annealing LR Schedule giúp hội tụ về flat minima
        
        # --- Độ chính xác & Tính tái lập khoa học (Scientific Reproducibility) ---
        amp=False,                     # FP32: Triệt tiêu sai số làm tròn float16, bảo toàn độ chính xác
        seed=42,                       # Cố định random seed để kết quả tái lập 100%
        deterministic=True,            # Bật thuật toán tất định cuDNN để mAP không bị trôi
        cache=False,                   # Tránh tràn RAM giữa các môi trường
        
        # --- Chiến lược Data Augmentation (Tối ưu cho ảnh tế bào vi mô BCCD) ---
        degrees=15.0,                  # Xoay ngẫu nhiên +/- 15 độ (tế bào xoay mọi hướng)
        fliplr=0.5,                    # Lật ngang xác suất 50%
        flipud=0.5,                    # Lật dọc xác suất 50% (rất tự nhiên với kính hiển vi)
        scale=0.5,                     # Đa tỉ lệ (0.5 - 1.5) nhận diện tế bào ở các cự ly
        mosaic=1.0,                    # Ghép 4 ảnh tăng mật độ và receptive field
        mixup=0.15,                    # Trộn ảnh nhẹ làm mềm biên tế bào chồng lấn
        close_mosaic=15,               # Tắt mosaic ở 15 epoch cuối (185-200) để tinh chỉnh biên thực
        
        # --- Đánh giá, Lưu trữ & Xuất biểu đồ phục vụ bài báo ---
        val=True,                      # Chạy validation mỗi epoch để đo mAP@50 và mAP@50:95
        patience=50,                   # Early stopping patience
        save=True,                     # Lưu checkpoint best.pt và last.pt
        save_period=10,                # Lưu checkpoint định kỳ mỗi 10 epoch
        plots=True,                    # Xuất PR curve, F1 curve, confusion matrix cho bài báo
        project='runs/train',          # Thư mục lưu kết quả thí nghiệm
        name='exp_bccd_adamw',         # Tên thí nghiệm
    )