"""Smoke test for YOLOv12 hybrid backbone variants.

Run from repo root:
    python test_yolo_v12_hybrid.py

Build từng YAML, in params/GFLOPs, forward (1,3,640,640) để bắt lỗi shape sớm.
"""

import os
import sys
import traceback

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ultralytics import YOLO  # noqa: E402


HYBRID_DIR = "ultralytics/cfg/models/v12/yolov12-Hybrid"

VARIANTS = [
    f"{HYBRID_DIR}/yolov12-MobileNetV4.yaml",
    f"{HYBRID_DIR}/yolov12-ResNet50.yaml",
    f"{HYBRID_DIR}/yolov12-GhostNet.yaml",
    f"{HYBRID_DIR}/yolov12-Ghost-MobileNetV4.yaml",
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

    try:
        info = model.info(detailed=False, verbose=False)
        if isinstance(info, tuple) and len(info) >= 2:
            n_layers, n_params = info[0], info[1]
            print(f"[INFO ] layers = {n_layers}")
            print(f"[INFO ] params = {human(n_params)}  ({n_params:,})")
            if len(info) >= 4:
                print(f"[INFO ] GFLOPs = {info[3]:.2f}")
    except Exception:
        n_params = sum(p.numel() for p in model.model.parameters())
        print(f"[INFO ] params (fallback) = {human(n_params)}  ({n_params:,})")

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
    import ultralytics
    print("Vendored ultralytics from:", ultralytics.__file__)
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
