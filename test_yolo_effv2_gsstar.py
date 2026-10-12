"""Smoke test: YOLOv11 + EfficientNetV2 backbone + GSStar neck.

Chạy từ repo root:
    python test_yolo_effv2_gsstar.py

Monkey-patch GSStarBlock & GSStarCSP vào ultralytics.nn.tasks namespace
trước khi YOLO(yaml) → parser tìm thấy class theo tên trong YAML mà không
phải sửa __init__.py / tasks.py.

So sánh:
    YOLOv11n baseline   vs   YOLOv11n-EffV2-GSStar
    → in params, GFLOPs, output shapes của cả hai để bạn quan sát
      mức độ "nhẹ hơn" của block mới.
"""

import os
import sys
import traceback

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Inject GSStar block vào tasks namespace -----------------------------
from ultralytics.nn.modules import gsstar_blocks as gs  # noqa: E402
import ultralytics.nn.tasks as tasks_mod                # noqa: E402
for name in gs.__all__:
    setattr(tasks_mod, name, getattr(gs, name))

from ultralytics import YOLO                            # noqa: E402


CANDIDATES = [
    ("baseline YOLOv11n",  "ultralytics/cfg/models/11/yolo11.yaml"),
    ("EffV2 + GSStar neck", "ultralytics/cfg/models/11/yolo11-EffV2-GSStar.yaml"),
]


def human(n):
    for u in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:7.2f}{u}"
        n /= 1000
    return f"{n:7.2f}T"


def smoke(name, yaml):
    print("\n" + "─" * 78)
    print(f"[BUILD] {name}  ←  {yaml}")
    print("─" * 78)
    if not os.path.exists(yaml):
        print(f"[SKIP] không tìm thấy {yaml}")
        return None
    try:
        model = YOLO(yaml)
    except Exception:
        print("[FAIL] build error:")
        traceback.print_exc()
        return None

    info = model.info(detailed=False, verbose=False)
    layers, params, gflops = None, None, None
    if isinstance(info, tuple):
        layers = info[0]
        params = info[1]
        if len(info) >= 4:
            gflops = info[3]
    else:
        params = sum(p.numel() for p in model.model.parameters())

    print(f"[INFO ] layers={layers}  params={human(params)} ({params:,})  "
          f"GFLOPs={gflops if gflops is not None else 'n/a'}")

    # forward dummy
    model.model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model.model(x)
    print(f"[FWD  ] OK — outputs: {_shape(y)}")
    return params, gflops


def _shape(y):
    if torch.is_tensor(y):
        return tuple(y.shape)
    if isinstance(y, (list, tuple)):
        return [_shape(yi) for yi in y]
    return type(y).__name__


def main():
    import ultralytics
    print("Vendored ultralytics:", ultralytics.__file__)
    print("Torch:", torch.__version__, " CUDA:", torch.cuda.is_available())

    results = {}
    for name, yp in CANDIDATES:
        results[name] = smoke(name, yp)

    print("\n" + "#" * 78)
    print("# SUMMARY")
    print("#" * 78)
    for name, r in results.items():
        if r is None:
            print(f"  [FAIL]  {name}")
        else:
            p, g = r
            print(f"  [PASS]  {name:<28}  params={human(p)}  "
                  f"GFLOPs={g if g is not None else '-'}")


if __name__ == "__main__":
    main()
