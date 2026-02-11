
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
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-test-11.yaml')
    model = YOLO(r'ultralytics/cfg/models/11/yolo11-EfficientNetV2.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-test-19.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-test-26.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/test/yolo11-bifpn-mhsa.yaml')
    # model = YOLO(r'ultralytics/cfg/models/v9/yolov9t.yaml')
    # model = YOLO(r'ultralytics/cfg/models/v10/yolov10n.yaml')

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