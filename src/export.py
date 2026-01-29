"""Export models to ONNX/TorchScript."""
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    parser.add_argument("--output", type=str, default="./exported")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Load model from checkpoint
    # TODO: Export to requested format

    print(f"Export to {args.format} not yet implemented")


if __name__ == "__main__":
    main()
