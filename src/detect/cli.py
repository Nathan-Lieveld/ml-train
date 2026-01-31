"""CLI entry point for the custom YOLO detection pipeline.

Usage::

    detect train  --model yolo11s --data lvis.yaml --epochs 100 --batch 16
    detect eval   --weights runs/detect/best.pt --data lvis.yaml
    detect export --weights best.pt --format onnx --imgsz 640
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

MODEL_CHOICES = ("yolov8s", "yolo11s", "yolo26s")
EXPORT_FORMATS = ("onnx", "torchscript")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detect", description="Custom YOLO detection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Train a detection model")
    p_train.add_argument("--model", type=str, default="yolo11s", choices=MODEL_CHOICES)
    p_train.add_argument("--data", type=str, required=True, help="Path to data YAML or COCO JSON")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=0.01)
    p_train.add_argument("--weights", type=str, default=None, help="Pretrained weights path")
    p_train.add_argument("--device", type=str, default=None, help="cuda or cpu")
    p_train.add_argument("--save-dir", type=str, default="runs/detect")
    p_train.add_argument("--accumulate", type=int, default=1)
    p_train.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    p_train.add_argument("--no-cache", action="store_true", help="Disable mmap cache")
    p_train.add_argument("--imgsz", type=int, default=640)

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Evaluate a detection model")
    p_eval.add_argument("--weights", type=str, required=True, help="Checkpoint path")
    p_eval.add_argument("--data", type=str, required=True, help="Path to data YAML or COCO JSON")
    p_eval.add_argument("--batch", type=int, default=16)
    p_eval.add_argument("--device", type=str, default=None)
    p_eval.add_argument("--imgsz", type=int, default=640)

    # --- export ---
    p_export = sub.add_parser("export", help="Export a detection model")
    p_export.add_argument("--weights", type=str, required=True, help="Checkpoint path")
    p_export.add_argument("--format", type=str, default="onnx", choices=EXPORT_FORMATS)
    p_export.add_argument("--imgsz", type=int, default=640)
    p_export.add_argument("--output", type=str, default="./exported")
    p_export.add_argument("--int8", action="store_true", help="Post-training INT8 quantization (ONNX only)")

    return parser


def _get_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(name: str, nc: int) -> torch.nn.Module:
    from src.detect.models import yolo11s, yolo26s, yolov8s

    constructors = {"yolov8s": yolov8s, "yolo11s": yolo11s, "yolo26s": yolo26s}
    return constructors[name](nc=nc)


def _load_data(data_path: str) -> tuple[object, int]:
    """Auto-detect data format and load annotations. Returns (DataFrame, nc)."""
    from src.detect.data import load_coco_annotations, load_yolo_annotations

    p = Path(data_path)
    if p.suffix == ".json":
        df = load_coco_annotations(p)
        nc = df.filter(df["cls"] >= 0)["cls"].n_unique() if not df.is_empty() else 80
    elif p.suffix in (".yaml", ".yml"):
        df = load_yolo_annotations(p)
        nc = df.filter(df["cls"] >= 0)["cls"].n_unique() if not df.is_empty() else 80
    else:
        raise ValueError(f"Unsupported data format: {p.suffix} (expected .json or .yaml)")
    return df, max(nc, 1)


def _cmd_train(args: argparse.Namespace) -> None:
    from src.detect.augment import Compose, LetterBox, RandomFlip, RandomHSV
    from src.detect.data import DiskImageStore, DetectionDataset, create_dataloader
    from src.detect.models import load_ultralytics_weights
    from src.detect.train import train

    df, nc = _load_data(args.data)
    device = _get_device(args.device)

    model = _build_model(args.model, nc)
    if args.weights:
        load_ultralytics_weights(model, args.weights)

    store = DiskImageStore(df)
    transforms = Compose([
        LetterBox(args.imgsz),
        RandomHSV(),
        RandomFlip(),
    ])
    dataset = DetectionDataset(df, store, transforms=transforms, imgsz=args.imgsz)
    train_loader = create_dataloader(dataset, batch_size=args.batch, shuffle=True)

    train(
        model,
        train_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
        batch_size=args.batch,
        accumulate=args.accumulate,
        resume=args.resume,
    )


def _cmd_eval(args: argparse.Namespace) -> None:
    from src.detect.augment import Compose, LetterBox
    from src.detect.data import DiskImageStore, DetectionDataset, create_dataloader
    from src.detect.eval import evaluate

    df, nc = _load_data(args.data)
    device = _get_device(args.device)

    model = _build_model("yolo11s", nc)
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    store = DiskImageStore(df)
    transforms = Compose([LetterBox(args.imgsz)])
    dataset = DetectionDataset(df, store, transforms=transforms, imgsz=args.imgsz)
    loader = create_dataloader(dataset, batch_size=args.batch, shuffle=False)

    metrics = evaluate(model, loader, device)
    print(f"mAP50: {metrics['mAP50']:.4f}  mAP50_95: {metrics['mAP50_95']:.4f}")


def _cmd_export(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)

    nc = ckpt.get("nc", 80)
    model = _build_model("yolo11s", nc)
    model.load_state_dict(ckpt["model"])
    model.eval()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)

    if args.format == "onnx":
        import onnx

        out_path = output_dir / "model.onnx"
        torch.onnx.export(
            model, dummy, str(out_path),
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        onnx.checker.check_model(str(out_path))
        logger.info("Exported ONNX: %s", out_path)

        if args.int8:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            int8_path = output_dir / "model_int8.onnx"
            quantize_dynamic(str(out_path), str(int8_path), weight_type=QuantType.QInt8)
            logger.info("Exported INT8 ONNX: %s", int8_path)
    elif args.format == "torchscript":
        out_path = output_dir / "model.torchscript"
        traced = torch.jit.trace(model, dummy)
        traced.save(str(out_path))
        logger.info("Exported TorchScript: %s", out_path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    commands = {"train": _cmd_train, "eval": _cmd_eval, "export": _cmd_export}
    commands[args.command](args)


if __name__ == "__main__":
    main()
