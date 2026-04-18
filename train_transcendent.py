"""Train YOLO-Transcendent.

Tối ưu cho Tesla P100 16GB / RTX 30xx-40xx tầm trung.
Chạy từ repo root:
    python train_transcendent.py
"""
from ultralytics import YOLO


def main():
    model = YOLO("ultralytics/cfg/models/11/yolo11-Transcendent/yolo11-Transcendent.yaml")
    model.info(detailed=False, verbose=True)

    model.train(
        data="coco.yaml",            # đổi sang data yaml của bạn
        imgsz=640,
        batch=16,                    # P100 16GB: an toàn với MSSO (loop scan)
        epochs=150,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,
        amp=True,
        device=0,
        workers=8,
        patience=30,
        seed=42,
        close_mosaic=10,
        project="runs/train",
        name="yolo11-transcendent",
    )


if __name__ == "__main__":
    main()
