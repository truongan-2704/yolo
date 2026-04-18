# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a YOLO research sandbox: a vendored copy of the Ultralytics repo (`ultralytics/`) plus a large collection of custom backbone / neck / attention experiments layered on top of it. Most work here is about inventing new architectures (Prism, Nexus, Chimera, Edge, Phoenix, Zenith, Spectra, SafeGuard, Hybrid crossovers, DCNF/StarFusion variants, MobileNetV4/EfficientNetV2 backbones, Transformer hybrids, YOLOv13) and comparing them against stock YOLOv8/11/12.

## Environment

- Python with dependencies pinned in `requirements.txt` (torch 2.2.2, torchvision 0.17.2, timm 1.0.22, numpy 1.26.4, opencv-python 4.9, albumentations 2.0.4, onnx/onnxruntime, supervision, einops, …). Install with `pip install -r requirements.txt`.
- No package install step — scripts at the repo root import the vendored `ultralytics/` package directly (`from ultralytics import YOLO`). Run everything from the repo root so that import path resolves.
- There is no `setup.py`/`pyproject.toml`, no lint config, and no pytest suite. The `test_*.py` files are smoke-test scripts, not pytest tests — run them with `python test_xxx.py`.

## Common commands

- Train the currently-selected architecture: `python train.py`
  - Model is chosen by uncommenting one `YOLO(...)` line near the top; training knobs (data yaml, imgsz, batch, epochs, optimizer, device, `project='runs/train'`) live in the `model.train(...)` call. Default here is CPU / FP32 / SGD — change `device='cpu'` and `amp=False` if running on GPU.
- Train the PPE SafeGuard recipe: `python train_ppe_safeguard.py` (AdamW + PPE-tuned augmentation, writes to `runs/train_ppe/safeguard`).
- Smoke-test a new architecture: pick the matching `test_yolo_<name>.py` (or `test_yolo13.py`, `test_dcnf_vX.py`, `test_yolo_hybrids.py`). They build the model from YAML, optionally exercise the custom blocks directly, run a dummy `torch.randn(1,3,640,640)` forward pass, and print param counts / output shapes.
- Compare param counts across configs: `python _check_params.py` (edit the `models` dict to the YAMLs you want to compare).
- Inference / PCB defect detection demo: `python check_PCB.py --weights <path.pt> --source <file-or-dir> [--imgsz 1024] [--classes short,spur] [--cross_class_nms] [--save_crops]`. Produces annotated images + `detections.csv` under `runs/pcb_detect/`.
- Vehicle counting demo: `python count_car.py`.

## Architecture map

### Vendored Ultralytics

`ultralytics/` is a full copy of the Ultralytics codebase (`cfg/`, `data/`, `engine/`, `hub/`, `models/`, `nn/`, `solutions/`, `trackers/`, `utils/`). The normal Ultralytics entry point `YOLO(<yaml-or-pt>)` is used unchanged — `.train()`, `.predict()`, `.info()` behave as documented upstream. Do not expect a pip-installed `ultralytics` — this local copy wins.

### Where custom architectures live

Two parallel trees, and both must stay in sync when adding a new block:

1. **Python implementations** under `ultralytics/nn/modules/`:
   - Stock upstream: `block.py`, `conv.py`, `head.py`, `transformer.py`, `activation.py`, `utils.py`.
   - Attention / backbone experiments: `CA.py`, `CBAM.py`, `ECA.py`, `EMA.py`, `GAM.py`, `SHSA.py`, `MHSA.py`, `SK.py`, `SimAM.py`, `ShuffleAttention.py`, `TripletAttention.py`, `AKConv.py`, `BiFPN.py`, `IDC.py`, `LAE.py`, `UNetV2.py`, `EfficientNetV2*.py`, `EfficientNetV4.py`, `MobileNetV4.py`, `MobileNetV4Pro.py`, `TransformerHybrid.py`.
   - Named custom architectures: `chimera_blocks.py`, `edge_blocks.py`, `hybrid_blocks.py`, `nexus_blocks.py`, `phoenix_blocks.py`, `prism_blocks.py`, `prism_v2_blocks.py`, `safeguard_blocks.py`, `spectra_blocks.py`, `yolo13_blocks.py`, `zenith_blocks.py`.
   - StarFusion / DCNF line: `starfusion_block.py`, `starfusion_block_v1plus.py`, `…_v2.py`…`_v6.py`.
   - Exports are aggregated in `ultralytics/nn/modules/__init__.py` — every new block class must be re-exported there so the YAML parser in `ultralytics/nn/tasks.py` can resolve it by name.

2. **YAML model configs** under `ultralytics/cfg/models/`:
   - Standard versions in `v3/ v5/ v6/ v8/ v9/ v10/ v12/ v13/` and `11/`.
   - Custom families each get their own subdirectory under `11/`: `yolo11-Chimera/`, `yolo11-EDGE/`, `yolo11-EffecientNetV2/`, `yolo11-Hybrid/`, `yolo11-MobileNetV4/`, `yolo11-MobileNetV4Pro/`, `yolo11-Nexus/`, `yolo11-Phoenix/`, `yolo11-Prism/`, `yolo11-SafeGuard/`, `yolo11-Spectra/`, `yolo11-TransformerHybrid/`, `yolo11-Zenith/`, plus many ablation YAMLs directly in `11/` (`yolo11-DCNF-V*.yaml`, `yolo11-EfficientNetV2-XX.yaml`, `yolo11-test-*.yaml`).
   - Design docs for each family are in `plans/` (`yolo-prism-architecture.md`, `yolo-safeguard-ppe-architecture.md`, `yolo-chimera-architecture.md`, `yolo-edge-architecture.md`, `yolo-spectra-architecture.md`, `yolo-prism-v2-architecture.md`, `yolo-hybrid-crossover-architectures.md`, `yolo13-architecture.md`, `efficientnetv2-dcnf-v5-optimization-plan.md`). Read these first before touching a family — they are the source of truth for the intended block composition.

### Adding a new custom block/architecture

End-to-end flow (consistent across every existing family):

1. Implement `nn.Module` classes in a new `ultralytics/nn/modules/<name>_blocks.py`.
2. Re-export the public classes from `ultralytics/nn/modules/__init__.py`.
3. Make sure `ultralytics/nn/tasks.py`'s `parse_model` can instantiate them by class name — that is how YAML `backbone:` / `head:` entries resolve.
4. Add a YAML at `ultralytics/cfg/models/11/yolo11-<Name>/yolo11-<Name>.yaml` (follow a sibling family as template).
5. Add a `test_yolo_<name>.py` at the repo root: build via `YOLO(yaml_path)`, optionally unit-check individual blocks with dummy tensors, then run a forward pass with `torch.randn(1,3,640,640)` and print `model.info()` + parameter count.
6. Wire it into `train.py` by adding a commented `YOLO(...)` line, then uncomment to train.

### Artifacts

- `runs/train/...` — default training output location used by `train.py`.
- `runs/train_ppe/safeguard` — PPE training output from `train_ppe_safeguard.py`.
- Pretrained/demo checkpoints at the repo root: `yolov8.pt`, `yolov8-bifpn-mhsa.pt`, `yolo11-bifpn-ca.pt`.
- `img*.png` at the repo root are sample images for demo scripts.

## Gotchas

- Scripts import the **vendored** `ultralytics/`. If a global `pip install ultralytics` is present it can shadow this — always run from the repo root and check `python -c "import ultralytics; print(ultralytics.__file__)"`.
- `train.py` as checked in targets CPU (`device='cpu'`, `amp=False`, `workers=0`) for debugging. Flip these for real training runs.
- Comments and some docstrings are in Vietnamese (e.g. `check_PCB.py`). Preserve them when editing.
- The `test_*.py` files are smoke/forward-pass harnesses, not a pytest suite — there is no `pytest` / lint / CI configured in this repo.