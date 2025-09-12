
from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-SHSA.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-unetv2.yaml')
    model = YOLO(r'ultralytics/cfg/models/v12/yolov12-bifpn-ca-02.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-p2-bifpn-mhsa.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-bifpn.yaml')

    model.train(
        data=r'data/coco_dataset/data.yaml',
        imgsz=640,
        batch=32,                         # Ổn định hơn, ít dao động mAP
        epochs=150,
        cache=False,
        amp=False,                        # FP32 cho độ chính xác cao nhất
        optimizer='SGD',
        patience=30,
        save_period=10,
        seed=42,
        project='runs/train',
        name='exp',
        workers=16,
        device='cpu',
        val=True
    )