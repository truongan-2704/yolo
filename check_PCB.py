# improved_pcb_detect.py
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
import glob
from typing import Dict, List, Tuple, Optional, Set

# ---------------- Utils ----------------
def ensure_image_list(source: str) -> list:
    p = Path(source)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    if p.is_dir():
        files = []
        for e in exts:
            files.extend(sorted(p.glob(e)))
        return [str(x) for x in files]
    if p.is_file():
        return [str(p)]
    # glob pattern
    files = []
    for e in ([source] if any(source.endswith(x.replace("*", "")) for x in exts) else [*(f"{source}/{e}" for e in exts)]):
        files.extend(glob.glob(e))
    return sorted(files)

def scale_draw_params(h: int):
    th = max(1, int(round(h / 300)))
    fs = 0.6 if h < 1000 else 0.8
    return th, fs

def draw_label(img, x, y, text, color, font_scale=0.6, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0

def cross_class_dedup(objs: List[dict], iou_thr: float = 0.6) -> List[dict]:
    objs = sorted(objs, key=lambda x: x["conf"], reverse=True)
    kept = []
    for o in objs:
        if all(iou_xyxy(o["bbox"], k["bbox"]) < iou_thr for k in kept):
            kept.append(o)
    return kept

# ---------------- Main ----------------
def detect_pcb(
    weights: str,
    source: str,
    out_dir: str = "runs/pcb_detect",
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.45,
    device: Optional[str | int] = None,
    half: bool = False,
    interest_classes: Optional[Set[str]] = None,
    cross_class_nms: bool = False,
    save_crops: bool = False
):
    """
    Detect PCB defects using a YOLO .pt model. Saves annotated images and detections.csv.
    - interest_classes: None -> lấy tất cả lớp trong model; hoặc set tên lớp cần (lowercase)
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    img_paths = ensure_image_list(source)
    if not img_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh từ source: {source}")

    model = YOLO(weights)

    run_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    out_root = Path(out_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Bảng kết quả tổng hợp
    rows = []

    # Bảng palette cho 6 lớp PCB (BGR)
    pcb_palette = {
        "missing_hole":     (46, 204, 113),
        "mouse_bite":       (231, 76, 60),
        "open_circuit":     (52, 152, 219),
        "short":            (241, 196, 15),
        "spur":             (155, 89, 182),
        "spurious_copper":  (52, 73, 94),
    }

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Bỏ qua (không đọc được): {img_path}")
            continue

        res = model.predict(
            img,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=(half and device != "cpu"),
            save=False,  # tự vẽ để kiểm soát màu/nhãn
            verbose=False,
            max_det=5000,
            save_crop=save_crops,
        )[0]

        names: Dict[int, str] = res.names  # id -> name
        # chuẩn hóa set lớp quan tâm
        if interest_classes is None:
            # dùng tất cả lớp trong model
            interest = {n.lower() for n in names.values()}
        else:
            interest = {c.lower() for c in interest_classes}

        det = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
                cls_name = names.get(cls_id, str(cls_id)).lower()
                if cls_name in interest:
                    det.append({
                        "cls_id": int(cls_id),
                        "cls_name": cls_name,
                        "conf": float(c),
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "image": Path(img_path).name,
                        "image_path": str(Path(img_path).resolve())
                    })

        if cross_class_nms:
            det = cross_class_dedup(det, iou_thr=0.6)

        # Vẽ
        h, w = img.shape[:2]
        thickness, font_scale = scale_draw_params(h)
        for idx, o in enumerate(det, 1):
            x1, y1, x2, y2 = map(int, o["bbox"])
            color = pcb_palette.get(o["cls_name"], (0, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            draw_label(img, x1, y1, f"{o['cls_name']} {o['conf']:.2f}", color, font_scale, thickness)

            # Ghi vào hàng CSV
            rows.append({
                "image": o["image"],
                "image_path": o["image_path"],
                "class_name": o["cls_name"],
                "class_id": o["cls_id"],
                "confidence": o["conf"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1
            })

        # Banner tóm tắt theo lớp
        counts = {}
        for o in det:
            counts[o["cls_name"]] = counts.get(o["cls_name"], 0) + 1
        summary = " | ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        if summary:
            txt = f"Total: {len(det)} | {summary}"
            cv2.putText(img, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale + 0.2, (0, 0, 0), thickness + 4)
            cv2.putText(img, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale + 0.2, (255, 255, 255), thickness + 1)

        # Lưu ảnh kết quả
        out_img = out_root / f"{Path(img_path).stem}_det.jpg"
        cv2.imwrite(str(out_img), img)
        print(f"[OK] {img_path} -> {out_img}")

    # Lưu CSV
    if rows:
        df = pd.DataFrame(rows)
        csv_path = out_root / "detections.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Lưu CSV: {csv_path}")
    else:
        print("[!] Không có bbox nào được phát hiện.")

    print(f"[DONE] Kết quả trong: {out_root}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("PCB Detection with YOLO (.pt)")
    ap.add_argument("--weights", type=str, required=True, help="ultralytics/test-pcb.pt")
    ap.add_argument("--source", type=str, required=True, help="cc1.jpg")
    ap.add_argument("--out", type=str, default="runs/pcb_detect", help="Thư mục xuất kết quả")
    ap.add_argument("--imgsz", type=int, default=1024, help="Kích thước suy luận (đối tượng nhỏ nên >=1024)")
    ap.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence")
    ap.add_argument("--iou", type=float, default=0.45, help="Ngưỡng NMS IoU")
    ap.add_argument("--device", type=str, default=None, help="'cpu' hoặc chỉ số GPU, vd: '0'")
    ap.add_argument("--half", action="store_true", help="Dùng FP16 trên GPU")
    ap.add_argument("--classes", type=str, default="", help="Danh sách lớp cần (phẩy), để trống = tất cả. vd: 'short,spur'")
    ap.add_argument("--cross_class_nms", action="store_true", help="Khử trùng lặp chéo lớp (IoU>0.6)")
    ap.add_argument("--save_crops", action="store_true", help="Lưu crop từng bbox (Ultralytics)")
    args = ap.parse_args()

    cls_set = None
    if args.classes.strip():
        cls_set = {c.strip().lower() for c in args.classes.split(",") if c.strip()}

    detect_pcb(
        weights=args.weights,
        source=args.source,
        out_dir=args.out,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        interest_classes=cls_set,
        cross_class_nms=args.cross_class_nms,
        save_crops=args.save_crops
    )
