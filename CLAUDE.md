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
uv run baseline train --data lvis.yaml --epochs 100 --batch 16

# On Windows, use --workers 0 to avoid multiprocessing issues
uv run baseline train --data lvis.yaml --epochs 100 --batch 16 --workers 0

# Export model (ONNX or TorchScript)
uv run export --checkpoint <path> --format onnx --output ./exported

# Run tests
uv run pytest

# Run single test
uv run pytest tests/test_models.py::test_forward_pass

# Lint
uv run ruff check src/ tests/
```

## Architecture

PyTorch-based ML training pipeline for edge deployment.

- **src/models.py**: Model definitions. `TinyConvNet` is a lightweight CNN (~50K params) designed for edge devices.
- **src/train.py**: Training entry point. Handles device selection (CUDA/CPU), optimizer setup, CLI args.
- **src/baseline_detection.py**: YOLO-based object detection pipeline (train, export, validate, benchmark). Uses ultralytics.
- **src/export.py**: Model export to ONNX/TorchScript for inference.
- **src/nas.py**: Neural architecture search with evolutionary optimization.
- **src/latency.py**: Latency lookup table builder for NAS.

Training flow: models.py (architecture) → train.py (training loop) → export.py (deployment artifacts)

## Platform Notes

- PyTorch CUDA 12.8 index is configured in `pyproject.toml` for GPU support on both Linux and Windows.
- On Windows, use `--workers 0` for baseline training to avoid multiprocessing issues.
