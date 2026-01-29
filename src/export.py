"""Export models to ONNX/TorchScript."""
import argparse
from pathlib import Path

import torch

from .models import TinyConvNet


def load_model_from_checkpoint(path: str, device: torch.device):
    """Load model from checkpoint and return in eval mode."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TinyConvNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def export_onnx(model, output_path: str):
    """Export model to ONNX format with dynamic batch axis."""
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported ONNX model to {output_path}")


def export_torchscript(model, output_path: str):
    """Export model to TorchScript via tracing."""
    dummy_input = torch.randn(1, 3, 32, 32)
    traced = torch.jit.trace(model, dummy_input)
    traced.save(output_path)
    print(f"Exported TorchScript model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", choices=["onnx", "torchscript", "both"], default="onnx")
    parser.add_argument("--output", type=str, default="./exported")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = load_model_from_checkpoint(args.checkpoint, device)

    if args.format in ("onnx", "both"):
        export_onnx(model, str(output_dir / "model.onnx"))

    if args.format in ("torchscript", "both"):
        export_torchscript(model, str(output_dir / "model.pt"))


if __name__ == "__main__":
    main()
