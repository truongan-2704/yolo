"""Smoke test cho khối C3k2_Titan (cải tiến C3k2 của YOLOv11).

Chạy từ repo root:
    python test_yolo_c3k2_titan.py

Điểm đặc biệt:
- Inject `C3k2_Titan` (+ phụ trợ) vào namespace `ultralytics.nn.tasks`
  bằng monkey-patch TRƯỚC khi gọi `YOLO(yaml)` → parser `parse_model`
  sẽ resolve được tên lớp mà không cần sửa `__init__.py` hay `tasks.py`.
- Sau khi build + forward, thử gọi `switch_to_deploy()` để fuse
  RepMultiKernelDW thành conv 7×7 deploy, rồi forward lại để xác nhận
  output khớp (sai số < 1e-4).
"""

import os
import sys
import traceback

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- B1: Import module Titan & inject vào tasks.py namespace ---------------
from ultralytics.nn.modules import c3k2_titan_blocks as titan  # noqa: E402
import ultralytics.nn.tasks as tasks_mod  # noqa: E402

for name in titan.__all__:
    setattr(tasks_mod, name, getattr(titan, name))

# --- B2: Giờ mới import YOLO để build model --------------------------------
from ultralytics import YOLO  # noqa: E402

YAML = "ultralytics/cfg/models/11/yolo11-Titan.yaml"


def human(n):
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:6.2f}{unit}"
        n /= 1000
    return f"{n:6.2f}T"


def main():
    import ultralytics
    print("Vendored ultralytics:", ultralytics.__file__)
    print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())

    print("\n[BUILD]", YAML)
    try:
        model = YOLO(YAML)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    # Params / GFLOPs
    try:
        info = model.info(detailed=False, verbose=False)
        if isinstance(info, tuple):
            print(f"[INFO ] layers = {info[0]}")
            print(f"[INFO ] params = {human(info[1])} ({info[1]:,})")
            if len(info) >= 4:
                print(f"[INFO ] GFLOPs = {info[3]:.2f}")
    except Exception:
        n_params = sum(p.numel() for p in model.model.parameters())
        print(f"[INFO ] params (fallback) = {human(n_params)}")

    # Forward train-mode
    model.model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y_train = model.model(x)
    print("[FWD  ] train-mode OK, output:", _shape(y_train))

    # --- B3: switch_to_deploy() — fuse RepMultiKernelDW ---------------------
    n_rep = 0
    for m in model.model.modules():
        if isinstance(m, titan.RepMultiKernelDW):
            m.switch_to_deploy()
            n_rep += 1
    print(f"[REP  ] fused {n_rep} RepMultiKernelDW blocks into single 7x7 DWConv")

    # Forward deploy-mode — phải ra kết quả GẦN BẰNG train-mode
    with torch.no_grad():
        y_deploy = model.model(x)
    print("[FWD  ] deploy-mode OK, output:", _shape(y_deploy))

    # Check sai số nếu outputs là tensor hoặc list of tensor
    max_err = _max_err(y_train, y_deploy)
    print(f"[DIFF ] max |train - deploy| = {max_err:.3e}")
    if max_err > 1e-3:
        print("[WARN ] sai số reparameterization hơi lớn — kiểm tra lại merge!")
    else:
        print("[OK   ] reparameterization numerically equivalent ✅")


def _shape(y):
    if torch.is_tensor(y):
        return tuple(y.shape)
    if isinstance(y, (list, tuple)):
        return [_shape(yi) for yi in y]
    return type(y).__name__


def _max_err(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return (a - b).abs().max().item()
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        errs = [_max_err(ai, bi) for ai, bi in zip(a, b)]
        errs = [e for e in errs if isinstance(e, float)]
        return max(errs) if errs else 0.0
    return 0.0


if __name__ == "__main__":
    main()
