
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
        workers=0,
        device='cpu',
        val=True
        # loss_type="aghiou",
    )