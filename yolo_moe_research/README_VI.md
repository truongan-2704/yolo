# MoE-YOLO: Spatial-Aware Dynamic Modality Routing

Đây là module PyTorch độc lập, được tạo ra để phục vụ cho bài báo nghiên cứu (CVPR/ICCV) của bạn.
Thiết kế của module nhằm đảm bảo **tuyệt đối không can thiệp hay chỉnh sửa bất kỳ source code nào của module Ultralytics cơ bản**.

## Cấu trúc
* `moe_yolo_block.py`: Chứa định nghĩa `MoEYOLOBlock` với 3 Experts (Light CNN, Dense CNN, Transformer) và hệ thống Router (Gating Network).

## Cách Tích Hợp Vào YOLOv8 (Mà Không Sửa Code Cũ)

Bước 1: Giữ nguyên source code YOLO hiện có.
Bước 2: Trong script train của bạn, hãy import và register module này vào thư viện ultralytics. Yolo hỗ trợ custom module cực tốt.

Hãy tạo một file `train_custom.py` của riêng bạn với nội dung tương tự sau:

```python
from ultralytics import YOLO
from ultralytics.nn.tasks import parse_model
import torch

# 1. Import module mới nghiên cứu
from yolo_moe_research.moe_yolo_block import MoEYOLOBlock

# 2. Bạn có thể chắp vá (Monkey-patch) thẻ module để parser của YOLO đọc được mà không cần sửa code trong folder siêu core `ultralytics/nn/modules`
import ultralytics.nn.modules as core_modules
core_modules.MoEYOLOBlock = MoEYOLOBlock

# 3. Tạo file cấu hình `yolov8-moe.yaml` tại thư mục ngoài cùng:
# (Copy từ yolov8.yaml, thay chữ C2f ở stage backbone bằng MoEYOLOBlock)

# 4. Train như bình thường
model = YOLO('yolov8-moe.yaml')
model.train(data='coco8.yaml', epochs=100, imgsz=640)
```

## Giải thích Code
Trong file `moe_yolo_block.py` ở hàm main (`__name__ == "__main__"`), đoạn mã đã tự động sinh một Tensor ngẫu nhiên và chạy thử, in ra kết quả qua khối Gate và các Experts để minh chứng cho việc forward logic hoàn toàn thành công và không lỗi memory leak.
