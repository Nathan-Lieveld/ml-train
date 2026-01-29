"""CoreML export for iOS deployment."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from .models import TinyConvNet
from .search_space import ArchConfig, SearchableNetwork


def export_coreml(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    quantize: bool = False,
    compute_units: str = "ALL",
) -> None:
    """Export PyTorch model to CoreML format.

    Args:
        model: PyTorch model to export (must be in eval mode)
        output_path: Path to save .mlpackage
        input_shape: Input tensor shape (batch, channels, height, width)
        quantize: If True, apply INT8 quantization
        compute_units: CoreML compute units ("ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE")
    """
    import coremltools as ct

    model.eval()

    # Trace the model
    dummy_input = torch.randn(*input_shape)
    traced = torch.jit.trace(model, dummy_input)

    # Map compute_units string to enum
    compute_units_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }

    # Convert to CoreML with FP16 precision
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=input_shape)],
        compute_units=compute_units_map.get(compute_units, ct.ComputeUnit.ALL),
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )

    # Apply quantization if requested
    if quantize:
        mlmodel = ct.compression_utils.affine_quantize_weights(mlmodel, mode="linear")

    # Save as mlpackage
    output_path = str(output_path)
    if not output_path.endswith(".mlpackage"):
        output_path = output_path + ".mlpackage"

    mlmodel.save(output_path)
    print(f"Exported CoreML model to {output_path}")


def export_coreml_flexible(
    model: torch.nn.Module,
    output_path: str,
    shapes: list[tuple[int, int, int, int]],
) -> None:
    """Export CoreML model with flexible input shapes using EnumeratedShapes.

    Args:
        model: PyTorch model to export
        output_path: Path to save .mlpackage
        shapes: List of valid input shapes [(batch, channels, height, width), ...]
    """
    import coremltools as ct

    model.eval()

    # Trace with first shape
    dummy_input = torch.randn(*shapes[0])
    traced = torch.jit.trace(model, dummy_input)

    # Create enumerated shapes for ANE optimization
    input_type = ct.TensorType(
        name="input",
        shape=ct.EnumeratedShapes(shapes=shapes),
    )

    mlmodel = ct.convert(
        traced,
        inputs=[input_type],
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )

    output_path = str(output_path)
    if not output_path.endswith(".mlpackage"):
        output_path = output_path + ".mlpackage"

    mlmodel.save(output_path)
    print(f"Exported CoreML model with flexible shapes to {output_path}")


def validate_coreml_model(model_path: str) -> dict:
    """Validate a CoreML model and return its specifications.

    Args:
        model_path: Path to .mlpackage

    Returns:
        Dictionary with input_specs, output_specs, and precision info
    """
    import coremltools as ct

    mlmodel = ct.models.MLModel(model_path)
    spec = mlmodel.get_spec()

    # Extract input specs
    input_specs = []
    for inp in spec.description.input:
        input_spec = {
            "name": inp.name,
            "type": inp.type.WhichOneof("Type"),
        }
        if inp.type.HasField("multiArrayType"):
            arr = inp.type.multiArrayType
            input_spec["shape"] = list(arr.shape)
            input_spec["dtype"] = str(arr.dataType)
        input_specs.append(input_spec)

    # Extract output specs
    output_specs = []
    for out in spec.description.output:
        output_spec = {
            "name": out.name,
            "type": out.type.WhichOneof("Type"),
        }
        if out.type.HasField("multiArrayType"):
            arr = out.type.multiArrayType
            output_spec["shape"] = list(arr.shape)
            output_spec["dtype"] = str(arr.dataType)
        output_specs.append(output_spec)

    # Check compute precision
    precision = "FP16" if spec.mlProgram else "FP32"

    return {
        "input_specs": input_specs,
        "output_specs": output_specs,
        "precision": precision,
        "compute_units": str(mlmodel.compute_unit),
    }


def load_model_from_checkpoint(
    path: str,
    device: torch.device,
    model_type: str = "tiny",
    config: ArchConfig | None = None,
) -> torch.nn.Module:
    """Load model from checkpoint.

    Args:
        path: Path to checkpoint file
        device: Device to load model on
        model_type: "tiny" for TinyConvNet, "searchable" for SearchableNetwork
        config: Required if model_type is "searchable"

    Returns:
        Model in eval mode
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model_type == "searchable":
        if config is None:
            # Try to load config from checkpoint
            if "config" in checkpoint:
                config = checkpoint["config"]
            else:
                raise ValueError("Config required for searchable model but not found in checkpoint")
        model = SearchableNetwork(config)
    else:
        model = TinyConvNet()

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    """CLI entry point for CoreML export."""
    parser = argparse.ArgumentParser(description="Export model to CoreML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output path for .mlpackage")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization")
    parser.add_argument(
        "--model-type",
        choices=["tiny", "searchable"],
        default="tiny",
        help="Model architecture type",
    )
    parser.add_argument(
        "--compute-units",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
        default="ALL",
        help="CoreML compute units",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = load_model_from_checkpoint(args.checkpoint, device, model_type=args.model_type)

    export_coreml(
        model,
        args.output,
        quantize=args.quantize,
        compute_units=args.compute_units,
    )


if __name__ == "__main__":
    main()
