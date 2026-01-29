"""Latency measurement and prediction for hardware-aware NAS."""
from __future__ import annotations

import argparse
import base64
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .search_space import (
    EXPANSIONS,
    KERNEL_SIZES,
    ArchConfig,
    SearchableBlock,
    SqueezeExcite,
)


@dataclass
class OpConfig:
    """Configuration for a single operation to benchmark."""

    op_type: str  # "conv", "dwconv", "se", "linear"
    channels: int
    kernel_size: int = 3
    expansion: int = 1
    use_se: bool = False
    stride: int = 1
    input_size: int = 32

    def to_key(self) -> str:
        """Convert to hashable string key."""
        return (
            f"{self.op_type}_c{self.channels}_k{self.kernel_size}"
            f"_e{self.expansion}_se{int(self.use_se)}_s{self.stride}_i{self.input_size}"
        )


@dataclass
class LatencyEntry:
    """Measurement result for an operation."""

    op_config: OpConfig
    mean_ms: float
    std_ms: float
    samples: int


class LatencyTable:
    """Lookup table for operation latencies."""

    def __init__(self):
        self._entries: dict[str, LatencyEntry] = {}

    def add(self, config: OpConfig, mean_ms: float, std_ms: float, samples: int) -> None:
        """Add a latency measurement to the table."""
        entry = LatencyEntry(
            op_config=config,
            mean_ms=mean_ms,
            std_ms=std_ms,
            samples=samples,
        )
        self._entries[config.to_key()] = entry

    def lookup(self, config: OpConfig) -> Optional[LatencyEntry]:
        """Look up latency for an operation config."""
        return self._entries.get(config.to_key())

    def save(self, path: str | Path) -> None:
        """Save table to JSON file."""
        data = []
        for entry in self._entries.values():
            data.append({
                "op_config": {
                    "op_type": entry.op_config.op_type,
                    "channels": entry.op_config.channels,
                    "kernel_size": entry.op_config.kernel_size,
                    "expansion": entry.op_config.expansion,
                    "use_se": entry.op_config.use_se,
                    "stride": entry.op_config.stride,
                    "input_size": entry.op_config.input_size,
                },
                "mean_ms": entry.mean_ms,
                "std_ms": entry.std_ms,
                "samples": entry.samples,
            })

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LatencyTable":
        """Load table from JSON file."""
        table = cls()
        with open(path) as f:
            data = json.load(f)

        for item in data:
            config = OpConfig(**item["op_config"])
            table.add(
                config,
                mean_ms=item["mean_ms"],
                std_ms=item["std_ms"],
                samples=item["samples"],
            )
        return table

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, config: OpConfig) -> bool:
        return config.to_key() in self._entries


def create_minimal_model(op_config: OpConfig) -> nn.Module:
    """Create a minimal model for benchmarking a single operation."""

    class MinimalModel(nn.Module):
        def __init__(self, op: nn.Module, in_channels: int):
            super().__init__()
            self.op = op
            self.in_channels = in_channels

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.op(x)

    if op_config.op_type == "block":
        # Full searchable block
        op = SearchableBlock(
            in_channels=op_config.channels,
            out_channels=op_config.channels * (2 if op_config.stride == 2 else 1),
            kernel_size=op_config.kernel_size,
            expansion=op_config.expansion,
            use_se=op_config.use_se,
            stride=op_config.stride,
        )
    elif op_config.op_type == "se":
        op = SqueezeExcite(op_config.channels)
    elif op_config.op_type == "conv":
        op = nn.Sequential(
            nn.Conv2d(op_config.channels, op_config.channels, op_config.kernel_size, padding=op_config.kernel_size // 2),
            nn.BatchNorm2d(op_config.channels),
            nn.ReLU(inplace=True),
        )
    elif op_config.op_type == "dwconv":
        op = nn.Sequential(
            nn.Conv2d(
                op_config.channels,
                op_config.channels,
                op_config.kernel_size,
                padding=op_config.kernel_size // 2,
                groups=op_config.channels,
            ),
            nn.BatchNorm2d(op_config.channels),
            nn.ReLU(inplace=True),
        )
    elif op_config.op_type == "stem":
        op = nn.Sequential(
            nn.Conv2d(3, op_config.channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(op_config.channels),
            nn.ReLU(inplace=True),
        )
        return MinimalModel(op, 3)
    elif op_config.op_type == "classifier":
        op = nn.Linear(op_config.channels, 10)
        return MinimalModel(op, op_config.channels)
    else:
        raise ValueError(f"Unknown op_type: {op_config.op_type}")

    return MinimalModel(op, op_config.channels)


def export_to_coreml_bytes(model: nn.Module, input_shape: tuple[int, ...]) -> bytes:
    """Export model to CoreML and return as bytes."""
    import coremltools as ct

    model.eval()
    dummy_input = torch.randn(*input_shape)
    traced = torch.jit.trace(model, dummy_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=input_shape)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.mlpackage"
        mlmodel.save(str(model_path))

        # Zip the mlpackage directory
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(model_path)
                    zf.write(file_path, arcname)

        return buffer.getvalue()


def measure_op_latency(
    op_config: OpConfig,
    device_ip: str,
    iterations: int = 100,
    port: int = 8765,
) -> LatencyEntry:
    """Measure latency of an operation on iOS device.

    Args:
        op_config: Operation configuration to benchmark
        device_ip: IP address of iOS device running BenchmarkApp
        iterations: Number of inference iterations
        port: HTTP server port on iOS device

    Returns:
        LatencyEntry with measurement results
    """
    import urllib.request

    # Create minimal model
    model = create_minimal_model(op_config)
    model.eval()

    # Determine input shape
    if op_config.op_type == "stem":
        input_shape = (1, 3, op_config.input_size, op_config.input_size)
    elif op_config.op_type == "classifier":
        input_shape = (1, op_config.channels)
    else:
        input_shape = (1, op_config.channels, op_config.input_size, op_config.input_size)

    # Export to CoreML
    model_bytes = export_to_coreml_bytes(model, input_shape)
    model_b64 = base64.b64encode(model_bytes).decode("ascii")

    # Send to iOS device
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

    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))

    return LatencyEntry(
        op_config=op_config,
        mean_ms=result["meanLatencyMs"],
        std_ms=result["stdLatencyMs"],
        samples=iterations,
    )


def estimate_network_latency(
    config: ArchConfig,
    lut: LatencyTable,
    input_size: int = 32,
) -> float:
    """Estimate total network latency using lookup table.

    Args:
        config: Network architecture configuration
        lut: Latency lookup table
        input_size: Input image size

    Returns:
        Estimated total latency in milliseconds
    """
    total_ms = 0.0
    current_size = input_size

    # Stem latency
    stem_config = OpConfig(
        op_type="stem",
        channels=config["base_channels"],
        input_size=current_size,
    )
    entry = lut.lookup(stem_config)
    if entry:
        total_ms += entry.mean_ms

    # Block latencies
    current_channels = config["base_channels"]
    for block_cfg in config["blocks"]:
        block_config = OpConfig(
            op_type="block",
            channels=current_channels,
            kernel_size=block_cfg["kernel_size"],
            expansion=block_cfg["expansion"],
            use_se=block_cfg["use_se"],
            stride=block_cfg["stride"],
            input_size=current_size,
        )
        entry = lut.lookup(block_config)
        if entry:
            total_ms += entry.mean_ms

        if block_cfg["stride"] == 2:
            current_size = current_size // 2
        current_channels = block_cfg["out_channels"]

    # Classifier latency (after global pooling, so spatial size = 1)
    classifier_config = OpConfig(
        op_type="classifier",
        channels=current_channels,
        input_size=1,
    )
    entry = lut.lookup(classifier_config)
    if entry:
        total_ms += entry.mean_ms

    return total_ms


def generate_lut_configs(
    base_channels_list: list[int] = [16, 24, 32],
    input_sizes: list[int] = [32, 16, 8],
) -> list[OpConfig]:
    """Generate all operation configs needed for LUT."""
    configs = []

    # Stem configs
    for base_ch in base_channels_list:
        configs.append(OpConfig(op_type="stem", channels=base_ch, input_size=32))

    # Block configs
    for base_ch in base_channels_list:
        for ch_mult in [1, 2, 4]:
            channels = base_ch * ch_mult
            for kernel in KERNEL_SIZES:
                for expansion in EXPANSIONS:
                    for use_se in [False, True]:
                        for stride in [1, 2]:
                            for input_size in input_sizes:
                                configs.append(
                                    OpConfig(
                                        op_type="block",
                                        channels=channels,
                                        kernel_size=kernel,
                                        expansion=expansion,
                                        use_se=use_se,
                                        stride=stride,
                                        input_size=input_size,
                                    )
                                )

    # Classifier configs
    for base_ch in base_channels_list:
        for ch_mult in [1, 2, 4]:
            channels = base_ch * ch_mult
            configs.append(OpConfig(op_type="classifier", channels=channels, input_size=1))

    return configs


def build_lut(
    device_ip: str,
    output_path: str,
    iterations: int = 100,
) -> LatencyTable:
    """Build complete latency lookup table.

    Args:
        device_ip: IP address of iOS device
        output_path: Path to save LUT JSON
        iterations: Iterations per measurement

    Returns:
        Populated LatencyTable
    """
    from tqdm import tqdm

    configs = generate_lut_configs()
    table = LatencyTable()

    # Try to load existing table for incremental updates
    output_file = Path(output_path)
    if output_file.exists():
        try:
            table = LatencyTable.load(output_path)
            print(f"Loaded existing LUT with {len(table)} entries")
        except Exception as e:
            print(f"Could not load existing LUT: {e}")

    # Filter configs not yet measured
    remaining = [c for c in configs if c not in table]
    print(f"Measuring {len(remaining)} operations ({len(configs) - len(remaining)} already in LUT)")

    for config in tqdm(remaining, desc="Building LUT"):
        try:
            entry = measure_op_latency(config, device_ip, iterations)
            table.add(config, entry.mean_ms, entry.std_ms, entry.samples)

            # Save incrementally
            table.save(output_path)
        except Exception as e:
            print(f"Failed to measure {config.to_key()}: {e}")

    return table


def main():
    """CLI entry point for building latency LUT."""
    parser = argparse.ArgumentParser(description="Build latency lookup table")
    parser.add_argument("--device-ip", type=str, required=True, help="iOS device IP address")
    parser.add_argument("--output", type=str, default="lut.json", help="Output path for LUT")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per measurement")
    args = parser.parse_args()

    table = build_lut(args.device_ip, args.output, args.iterations)
    print(f"Built LUT with {len(table)} entries, saved to {args.output}")


if __name__ == "__main__":
    main()
