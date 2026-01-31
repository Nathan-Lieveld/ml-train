"""Baseline YOLO object detection for iPhone deployment."""
from __future__ import annotations

import argparse
import base64
import json
import platform
import urllib.request
from pathlib import Path

from ultralytics import YOLO

_IS_WINDOWS = platform.system() == "Windows"
_DEFAULT_WORKERS = 0 if _IS_WINDOWS else 4


def train_model(
    model_name: str = "yolo11n.pt",
    data_yaml: str = "lvis.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str | None = None,
    freeze: int | None = None,
    device: str = "0",
    workers: int = _DEFAULT_WORKERS,
    cache: str | bool = "disk",
    deterministic: bool = False,
    resume: str | None = None,
) -> Path:
    """Fine-tune a YOLO model on a dataset.

    Args:
        model_name: Pretrained YOLO checkpoint (e.g. yolo11n.pt, yolov8s.pt)
        data_yaml: Path or name of dataset YAML config (e.g. lvis.yaml, coco.yaml)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        project: Output project directory
        name: Experiment name (defaults to model stem + dataset stem)
        freeze: Number of backbone layers to freeze (None = train all)
        device: CUDA device (e.g. '0', '0,1', 'cpu')
        workers: Number of dataloader workers (auto: 0 on Windows, 4 on Linux)
        cache: Cache mode - 'disk' (save decoded .npy), True/'ram' (in-memory), or False
        deterministic: Use deterministic algorithms (slower but reproducible)
        resume: Path to last.pt checkpoint to resume training from

    Returns:
        Path to best checkpoint
    """
    if resume:
        model = YOLO(resume)
        model.train(resume=True)
    else:
        if name is None:
            model_stem = Path(model_name).stem
            data_stem = Path(data_yaml).stem
            name = f"{model_stem}_{data_stem}"

        model = YOLO(model_name)

        train_kwargs = dict(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            device=device,
            workers=workers,
            cache=cache,
            deterministic=deterministic,
            patience=20,
            save=True,
            plots=True,
        )
        if freeze is not None:
            train_kwargs["freeze"] = freeze

        model.train(**train_kwargs)

    best_path = Path(project) / name / "weights" / "best.pt"
    print(f"Training complete. Best model: {best_path}")
    return best_path


def export_model(
    checkpoint: str,
    format: str = "coreml",
    imgsz: int = 640,
    half: bool = True,
    nms: bool = True,
) -> Path:
    """Export YOLOv8 model for deployment.

    Supported formats:
    - coreml: CoreML for iOS (requires macOS for full export)
    - onnx: ONNX format (works cross-platform, can convert to CoreML on macOS)

    Args:
        checkpoint: Path to YOLOv8 checkpoint (.pt)
        format: Export format ('coreml' or 'onnx')
        imgsz: Input image size
        half: Use FP16 precision (recommended for ANE)
        nms: Include NMS in model

    Returns:
        Path to exported model
    """
    model = YOLO(checkpoint)

    export_path = model.export(
        format=format,
        imgsz=imgsz,
        half=half,
        nms=nms,
    )

    print(f"Exported {format.upper()} model: {export_path}")
    return Path(export_path)


def benchmark_on_device(
    checkpoint: str,
    device_ip: str,
    iterations: int = 100,
    port: int = 8765,
) -> dict:
    """Benchmark model latency on iOS device.

    Requires BenchmarkApp running on iPhone (sideload via GitHub Actions build).

    Args:
        checkpoint: Path to YOLOv8 checkpoint (.pt)
        device_ip: IP address of iPhone running BenchmarkApp
        iterations: Number of inference iterations
        port: BenchmarkApp server port

    Returns:
        Dict with latency stats (meanLatencyMs, stdLatencyMs, etc.)
    """
    import tempfile

    import coremltools as ct

    model = YOLO(checkpoint)

    # Export to CoreML using neuralnetwork format (works on Linux)
    print("Exporting model to CoreML (neuralnetwork format)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # First export to ONNX
        onnx_path = model.export(format="onnx", imgsz=640, half=False)

        # Then convert ONNX to CoreML neuralnetwork format
        import onnx

        onnx_model = onnx.load(onnx_path)

        mlmodel = ct.converters.onnx.convert(
            onnx_model,
            minimum_ios_deployment_target="16.0",
        )

        model_path = Path(tmpdir) / "model.mlmodel"
        mlmodel.save(str(model_path))
        model_bytes = model_path.read_bytes()

    model_b64 = base64.b64encode(model_bytes).decode("ascii")

    # Send to iOS device
    print(f"Sending model to {device_ip}:{port} for benchmarking...")
    url = f"http://{device_ip}:{port}/benchmark"
    payload = json.dumps({
        "modelData": model_b64,
        "iterations": iterations,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    # Calculate FPS
    mean_ms = result["meanLatencyMs"]
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0

    print("\nBenchmark Results:")
    print(f"  Mean latency: {mean_ms:.2f} ms")
    print(f"  Std latency:  {result['stdLatencyMs']:.2f} ms")
    print(f"  Min latency:  {result['minLatencyMs']:.2f} ms")
    print(f"  Max latency:  {result['maxLatencyMs']:.2f} ms")
    print(f"  Estimated FPS: {fps:.1f}")
    print(f"  Target (30 FPS): {'PASS' if fps >= 30 else 'FAIL'}")

    return {**result, "fps": fps}


def validate_model(checkpoint: str, data_yaml: str) -> dict:
    """Validate model on test set.

    Args:
        checkpoint: Path to model checkpoint
        data_yaml: Path to dataset config

    Returns:
        Validation metrics dict
    """
    model = YOLO(checkpoint)
    metrics = model.val(data=data_yaml)
    return {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO baseline for iPhone detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Fine-tune a YOLO model")
    train_parser.add_argument("--model", type=str, default="yolo11n.pt",
                              help="Pretrained model (default: yolo11n.pt)")
    train_parser.add_argument("--data", type=str, default="lvis.yaml",
                              help="Dataset YAML (default: lvis.yaml)")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--name", type=str, default=None)
    train_parser.add_argument("--freeze", type=int, default=None,
                              help="Number of backbone layers to freeze")
    train_parser.add_argument("--device", type=str, default="0",
                              help="CUDA device (default: '0', use 'cpu' for CPU)")
    train_parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS,
                              help=f"Dataloader workers (default: {_DEFAULT_WORKERS})")
    train_parser.add_argument("--cache", type=str, default="disk",
                              help="Cache mode: 'disk', 'ram', or 'false' (default: disk)")
    train_parser.add_argument("--deterministic", action="store_true",
                              help="Use deterministic algorithms (slower)")
    train_parser.add_argument("--resume", type=str, default=None,
                              help="Path to last.pt checkpoint to resume training")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--checkpoint", type=str, required=True)
    export_parser.add_argument("--format", type=str, default="onnx", choices=["coreml", "onnx"],
                               help="Export format (onnx recommended on Linux, coreml on macOS)")
    export_parser.add_argument("--imgsz", type=int, default=640)
    export_parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    export_parser.add_argument("--no-nms", action="store_true", help="Exclude NMS from model")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate model")
    val_parser.add_argument("--checkpoint", type=str, required=True)
    val_parser.add_argument("--data", type=str, required=True)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark on iPhone")
    bench_parser.add_argument("--checkpoint", type=str, default="yolov8n.pt",
                              help="Model checkpoint (default: yolov8n.pt)")
    bench_parser.add_argument("--device-ip", type=str, required=True,
                              help="iPhone IP address running BenchmarkApp")
    bench_parser.add_argument("--iterations", type=int, default=100)
    bench_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()

    if args.command == "train":
        cache_val: str | bool = args.cache
        if cache_val.lower() == "false":
            cache_val = False
        elif cache_val.lower() in ("true", "ram"):
            cache_val = True

        train_model(
            model_name=args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name,
            freeze=args.freeze,
            device=args.device,
            workers=args.workers,
            cache=cache_val,
            deterministic=args.deterministic,
            resume=args.resume,
        )
    elif args.command == "export":
        export_model(
            checkpoint=args.checkpoint,
            format=args.format,
            imgsz=args.imgsz,
            half=not args.fp32,
            nms=not args.no_nms,
        )
    elif args.command == "validate":
        metrics = validate_model(args.checkpoint, args.data)
        print("Validation Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    elif args.command == "benchmark":
        benchmark_on_device(
            checkpoint=args.checkpoint,
            device_ip=args.device_ip,
            iterations=args.iterations,
            port=args.port,
        )


if __name__ == "__main__":
    main()
