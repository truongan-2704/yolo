from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load model YOLOv8
# model = YOLO('yolov8.pt')
model = YOLO('yolov8-bifpn-mhsa.pt')

# Đọc ảnh
image = cv2.imread('img_4.png')

# Predict
results = model(image)

# Class ID COCO
CAR_CLASS_ID = 1
MOTORBIKE_CLASS_ID = 2
BUS_CLASS_ID = 0
TRUCK_CLASS_ID = 3

# Lưu object detect
detected_objects = []
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0]

    detected_objects.append({
        "cls_id": cls_id,
        "conf": conf,
        "bbox": (x1, y1, x2, y2)
    })

# IoU
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2

    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2p - x1p) * (y2p - y1p)

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# Lọc trùng (NMS thủ công)
filtered_objects = []
while detected_objects:
    best = max(detected_objects, key=lambda x: x["conf"])
    detected_objects = [
        obj for obj in detected_objects
        if iou(obj["bbox"], best["bbox"]) < 0.5
    ]
    filtered_objects.append(best)

# Phân loại
cars = [o for o in filtered_objects if o["cls_id"] == CAR_CLASS_ID]
motorbikes = [o for o in filtered_objects if o["cls_id"] == MOTORBIKE_CLASS_ID]
buses = [o for o in filtered_objects if o["cls_id"] == BUS_CLASS_ID]
trucks = [o for o in filtered_objects if o["cls_id"] == TRUCK_CLASS_ID]

total = len(cars) + len(motorbikes) + len(buses) + len(trucks)

print(f"Car: {len(cars)}")
print(f"Motorbike: {len(motorbikes)}")
print(f"Bus: {len(buses)}")
print(f"Truck: {len(trucks)}")
print(f"Total vehicles: {total}")

# Vẽ box (TO + RÕ)
def draw_boxes(img, boxes, label, color):
    for i, obj in enumerate(boxes):
        x1, y1, x2, y2 = obj["bbox"]
        conf = obj["conf"]

        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness=3
        )

        text = f"{label}: {conf:.2f}"
        cv2.putText(
            img,
            text,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

# Draw
draw_boxes(image, cars, "Car", (0, 255, 0))
draw_boxes(image, motorbikes, "Motorbike", (255, 255, 0))
draw_boxes(image, buses, "Bus", (255, 0, 0))
draw_boxes(image, trucks, "Truck", (0, 0, 255))

# Tổng số
cv2.putText(
    image,
    f"Total Vehicles: {total}",
    (10, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (0, 255, 255),
    3
)

# Show
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
