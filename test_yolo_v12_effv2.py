"""Smoke test for YOLOv12 + EfficientNetV2 hybrid variants.

Run from repo root:
    python test_yolo_v12_effv2.py

Mục đích:
- Build từng YAML qua YOLO(yaml_path).
- In số tham số / GFLOPs / số layer.
- Forward 1 ảnh giả (1, 3, 640, 640) để chắc chắn không lỗi shape ở Concat/Detect.

KHÔNG sai sót: nếu một biến thể fail, script in stack trace nhưng vẫn chạy
tiếp các biến thể còn lại để tiện debug đối chiếu.
"""

import os
import sys
import traceback

import torch

# Đảm bảo import vendored ultralytics (chạy từ repo root)
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ultralytics import YOLO  # noqa: E402


VARIANTS = [
    "ultralytics/cfg/models/v12/yolov12-EfficientNetV2.yaml",
    "ultralytics/cfg/models/v12/yolov12-EfficientNetV2-P2.yaml",
    "ultralytics/cfg/models/v12/yolov12-EfficientNetV2-CA.yaml",
]


def human(n):
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:6.2f}{unit}"
        n /= 1000
    return f"{n:6.2f}T"


def smoke_one(yaml_path: str):
    print("\n" + "=" * 78)
    print(f"[BUILD] {yaml_path}")
    print("=" * 78)
    try:
        model = YOLO(yaml_path)
    except Exception:
        print("[FAIL] YOLO(yaml) ném exception:")
        traceback.print_exc()
        return False

    # Tổng kết model
    try:
        info = model.info(detailed=False, verbose=False)
        # info trả về (n_layers, n_params, n_grads, gflops) trên Ultralytics mới
        if isinstance(info, tuple) and len(info) >= 2:
            n_params = info[1]
            print(f"[INFO ] params = {human(n_params)}  ({n_params:,})")
            if len(info) >= 4:
                print(f"[INFO ] GFLOPs = {info[3]:.2f}")
            print(f"[INFO ] layers = {info[0]}")
    except Exception:
        n_params = sum(p.numel() for p in model.model.parameters())
        print(f"[INFO ] params (fallback) = {human(n_params)}  ({n_params:,})")

    # Forward dummy
    try:
        model.model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            y = model.model(x)
        if isinstance(y, (list, tuple)):
            shapes = []
            for yi in y:
                if torch.is_tensor(yi):
                    shapes.append(tuple(yi.shape))
                elif isinstance(yi, (list, tuple)):
                    shapes.append([tuple(t.shape) for t in yi if torch.is_tensor(t)])
            print(f"[FWD  ] OK — outputs: {shapes}")
        else:
            print(f"[FWD  ] OK — output shape: {tuple(y.shape)}")
        return True
    except Exception:
        print("[FAIL] forward(1,3,640,640) ném exception:")
        traceback.print_exc()
        return False


def main():
    print("Vendored ultralytics from:", end=" ")
    import ultralytics
    print(ultralytics.__file__)
    print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())

    results = {}
    for yp in VARIANTS:
        if not os.path.exists(yp):
            print(f"[SKIP] không tìm thấy {yp}")
            results[yp] = False
            continue
        results[yp] = smoke_one(yp)

    print("\n" + "#" * 78)
    print("# SUMMARY")
    print("#" * 78)
    for k, v in results.items():
        flag = "PASS" if v else "FAIL"
        print(f"  [{flag}] {k}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
