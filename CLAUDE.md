# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development

Use `uv` for all dependency management (never pip).

```bash
# Install dependencies (pulls CUDA 12.8 PyTorch wheels automatically)
uv sync --all-extras

# Run training (TinyConvNet on CIFAR-10)
uv run train --epochs 10 --lr 0.001

# Run YOLO baseline training (LVIS fine-tune)
# Defaults: --cache disk, --workers auto (0 on Windows, 4 on Linux)
uv run baseline train --data lvis.yaml --epochs 100 --batch 16

# Disable caching if disk space is limited
uv run baseline train --data lvis.yaml --epochs 100 --batch 16 --cache false

# Export model (ONNX or TorchScript)
uv run export --checkpoint <path> --format onnx --output ./exported

# Custom YOLO detection pipeline (train, eval, export)
uv run detect train --model yolo11s --data data/visdrone/data.yaml --epochs 100 --batch 16
uv run detect eval --weights runs/detect/best.pt --data data/visdrone/data.yaml
uv run detect export --weights runs/detect/best.pt --format onnx

# Run tests
uv run pytest

# Run single test
uv run pytest tests/test_models.py::test_forward_pass

# Lint
uv run ruff check src/ tests/ scripts/
```

## Architecture

PyTorch-based ML training pipeline for edge deployment.

- **src/models.py**: Model definitions. `TinyConvNet` is a lightweight CNN (~50K params) designed for edge devices.
- **src/train.py**: Training entry point. Handles device selection (CUDA/CPU), optimizer setup, CLI args.
- **src/baseline_detection.py**: YOLO-based object detection pipeline (train, export, validate, benchmark). Uses ultralytics.
- **src/detect/**: Custom YOLO detection pipeline (replaces ultralytics dependency for training).
  - `models.py`: YOLO architectures (yolov8s, yolo11s, yolo26s) with ultralytics weight loading.
  - `train.py`: Training loop with ModelEMA, SGD + cosine LR, warmup, AMP, checkpointing.
  - `loss.py`: Detection loss (CIoU + DFL + BCE cls) with task-aligned assigner.
  - `eval.py`: NMS, per-class AP, mAP evaluation.
  - `data.py`: Annotation parsing (YOLO/COCO formats), mmap image cache, Dataset/DataLoader.
  - `augment.py`: Detection augmentations (LetterBox, RandomHSV, RandomFlip, Mosaic).
  - `cli.py`: CLI entry point with `train`, `eval`, `export` subcommands.
- **src/export.py**: Model export to ONNX/TorchScript for inference.
- **src/nas.py**: Neural architecture search with evolutionary optimization.
- **src/latency.py**: Latency lookup table builder for NAS.
- **scripts/**: Dataset preparation utilities (`prepare_visdrone.py`, `gen_synthetic_detect.py`).

Training flow: models.py (architecture) → train.py (training loop) → export.py (deployment artifacts)
Detection flow: detect/models.py (architecture) → detect/train.py (training) → detect/cli.py export (deployment)

## Platform Notes

- PyTorch CUDA 12.8 index is configured in `pyproject.toml` for GPU support on both Linux and Windows.
- Workers default to 0 on Windows (multiprocessing issues) and 4 on Linux/WSL2 automatically.
- For best training throughput, use WSL2 (Ubuntu) — enables multiprocessing workers and native ext4 I/O.
- Disk caching (`--cache disk`) is enabled by default; first epoch writes `.npy` files, subsequent epochs skip JPEG decoding.
