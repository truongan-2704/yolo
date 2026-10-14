import sys
import torch
import yaml
from pathlib import Path
from ultralytics.models.yolo import YOLO
from ultralytics.nn.tasks import DetectionModel

def run_tests():
    configs = [
        ("Baseline (No DFDC, No MCG, No FASR)", "ultralytics/cfg/models/11/yolo11-Prism/ablation/yolo11-Prism-baseline.yaml"),
        ("DFDC Only", "ultralytics/cfg/models/11/yolo11-Prism/ablation/yolo11-Prism-only-DFDC.yaml"),
        ("MCG Only", "ultralytics/cfg/models/11/yolo11-Prism/ablation/yolo11-Prism-only-MCG.yaml"),
        ("FASR Only", "ultralytics/cfg/models/11/yolo11-Prism/ablation/yolo11-Prism-only-FASR.yaml"),
        ("w/o FASR (DFDC + MCG)", "ultralytics/cfg/models/11/yolo11-Prism/ablation/yolo11-Prism-wo-FASR.yaml"),
        ("w/o MCG (DFDC + FASR)", "ultralytics/cfg/models/11/yolo11-Prism/ablation/yolo11-Prism-wo-MCG.yaml"),
        ("w/o DFDC (MCG + FASR)", "ultralytics/cfg/models/11/yolo11-Prism/ablation/yolo11-Prism-wo-DFDC.yaml"),
        ("w/o High-Pass (No-HP)", "ultralytics/cfg/models/11/yolo11-Prism/ablation/yolo11-Prism-wo-HP.yaml"),
        ("Full YOLO11-Prism", "ultralytics/cfg/models/11/yolo11-Prism/yolo11-Prism.yaml"),
    ]

    x = torch.zeros(1, 3, 640, 640)

    for scale in ["n", "s"]:
        scale_name = "NANO (Scale 'n')" if scale == "n" else "SMALL (Scale 's')"
        print("\n" + "=" * 90)
        print(f" TESTING {scale_name}")
        print("=" * 90)
        print(f"{'Variant':<35} | {'Params (M)':<12} | {'Forward Shape':<25} | {'Status'}")
        print("-" * 90)

        for name, cfg_path in configs:
            p = Path(cfg_path)
            if not p.exists():
                print(f"{name:<35} | {'MISSING FILE':<12} | {'-':<25} | FAILED")
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    d = yaml.safe_load(f)
                d["scale"] = scale
                model = DetectionModel(d, nc=80, verbose=False)
                y = model(x)
                params = sum(param.numel() for param in model.parameters()) / 1e6
                if isinstance(y, (list, tuple)):
                    shape_str = str(y[0].shape if hasattr(y[0], "shape") else len(y))
                else:
                    shape_str = str(y.shape)
                print(f"{name:<35} | {params:<12.4f} | {shape_str:<25} | PASS")
            except Exception as e:
                print(f"{name:<35} | {'ERROR':<12} | {str(e)[:25]:<25} | FAILED")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 90)

if __name__ == "__main__":
    run_tests()
