from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Tải mô hình YOLO
model = YOLO('yolov8_bifpn.pt')

# Đọc ảnh đầu vào
image_path = 'img.png'
image = cv2.imread(image_path)

# Dự đoán đối tượng
results = model(image)

# Class ID trong dataset COCO
CAR_CLASS_ID = 1  # Xe hơi
TRUCK_CLASS_ID = 3  # Xe tải
BUS_CLASS_ID = 0  # Xe buýt

# Tạo danh sách chứa bounding boxes
detected_objects = []
for box in results[0].boxes:
    cls_id = int(box.cls[0])  # Lấy ID class
    conf = float(box.conf[0])  # Lấy độ tin cậy
    x1, y1, x2, y2 = box.xyxy[0]  # Lấy tọa độ

    detected_objects.append({
        "cls_id": cls_id,
        "conf": conf,
        "bbox": (x1, y1, x2, y2)
    })


# Hàm tính IoU (Intersection over Union)
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


# Lọc trùng lặp dựa trên IoU và độ tin cậy
filtered_objects = []
while detected_objects:
    best_obj = max(detected_objects, key=lambda x: x["conf"])  # Lấy object có conf cao nhất
    detected_objects = [obj for obj in detected_objects if iou(obj["bbox"], best_obj["bbox"]) < 0.5]  # Xóa obj trùng
    filtered_objects.append(best_obj)

# Phân loại lại danh sách
car_boxes = [obj for obj in filtered_objects if obj["cls_id"] == CAR_CLASS_ID]
truck_boxes = [obj for obj in filtered_objects if obj["cls_id"] == TRUCK_CLASS_ID]
bus_boxes = [obj for obj in filtered_objects if obj["cls_id"] == BUS_CLASS_ID]

# Đếm số lượng
num_cars = len(car_boxes)
num_trucks = len(truck_boxes)
num_buses = len(bus_boxes)
total_vehicles = num_cars + num_trucks + num_buses

print(f"Số lượng xe ô tô: {num_cars}")
print(f"Số lượng xe tải: {num_trucks}")
print(f"Số lượng xe buýt: {num_buses}")
print(f"Tổng số phương tiện: {total_vehicles}")


# Hàm vẽ bounding boxes
def draw_boxes(image, boxes, label_prefix, color):
    for i, obj in enumerate(boxes):
        x1, y1, x2, y2 = obj["bbox"]
        conf = obj["conf"]

        # Vẽ hình chữ nhật
        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

        # Hiển thị nhãn
        label = f"{label_prefix} {i + 1}: {conf:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# Vẽ bounding boxes cho từng loại
draw_boxes(image, car_boxes, "Car", (0, 255, 0))  # Xanh lá
draw_boxes(image, truck_boxes, "Truck", (0, 0, 255))  # Đỏ
draw_boxes(image, bus_boxes, "Bus", (255, 0, 0))  # Xanh dương

# Hiển thị tổng số lượng phương tiện
summary_label = f"Total Vehicles: {total_vehicles}"
cv2.putText(image, summary_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

# Hiển thị ảnh với Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
