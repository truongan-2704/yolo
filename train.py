
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO(r'ultralytics/cfg/models/v8/yolov8-simam.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-p2-bifpn-mhsa.yaml')
    # model = YOLO(r'ultralytics/cfg/models/11/yolo11-SAHI.yaml')
    model = YOLO(r'ultralytics/cfg/models/11/yolo11-GAM.yaml')



    model.train(
        data=r'data/coco_dataset/data.yaml',
        cache=False,
        imgsz=640,
        epochs= 2,
        single_cls=False,
        # batch=2,
        # close_mosaic=0,
        # worker=0,
        device='cpu',
        amp=False,
        project='runs/train',
        name='exp'
    )