"""Searchable architecture space for neural architecture search."""
from __future__ import annotations

import copy
import random
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn


class BlockConfig(TypedDict):
    """Configuration for a single searchable block."""

    kernel_size: int  # 3 or 5
    expansion: int  # 2, 4, or 6
    use_se: bool
    stride: int  # 1 or 2
    out_channels: int


class ArchConfig(TypedDict):
    """Configuration for a searchable network architecture."""

    num_blocks: int  # 2-6
    base_channels: int  # 16, 24, or 32
    blocks: list[BlockConfig]


# Valid choices for each parameter
KERNEL_SIZES = [3, 5]
EXPANSIONS = [2, 4, 6]
BASE_CHANNELS = [16, 24, 32]
CHANNEL_MULTIPLIERS = [1, 2, 4]  # For block out_channels relative to base


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation channel attention module."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SearchableBlock(nn.Module):
    """Inverted residual block with configurable kernel, expansion, and SE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        expansion: int = 4,
        use_se: bool = False,
        stride: int = 1,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expansion
        padding = kernel_size // 2

        layers: list[nn.Module] = []

        # 1x1 expand (skip if expansion == 1)
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])

        # Depthwise conv
        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        ])

        # SE module (optional)
        if use_se:
            layers.append(SqueezeExcite(hidden_dim))

        # 1x1 project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


class SearchableNetwork(nn.Module):
    """Network composed of searchable blocks from a configuration."""

    def __init__(
        self,
        config: ArchConfig,
        num_classes: int = 10,
        input_size: int = 32,
    ):
        super().__init__()
        self.config = config
        self.input_size = input_size

        base_ch = config["base_channels"]

        # Stem: 3x3 conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        # Build blocks from config
        blocks: list[nn.Module] = []
        in_channels = base_ch
        for block_cfg in config["blocks"]:
            block = SearchableBlock(
                in_channels=in_channels,
                out_channels=block_cfg["out_channels"],
                kernel_size=block_cfg["kernel_size"],
                expansion=block_cfg["expansion"],
                use_se=block_cfg["use_se"],
                stride=block_cfg["stride"],
            )
            blocks.append(block)
            in_channels = block_cfg["out_channels"]

        self.blocks = nn.Sequential(*blocks)
        self.final_channels = in_channels

        # Global pool and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.final_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def sample_random_config() -> ArchConfig:
    """Sample a random valid architecture configuration."""
    num_blocks = random.randint(2, 6)
    base_channels = random.choice(BASE_CHANNELS)

    blocks: list[BlockConfig] = []
    current_channels = base_channels

    for i in range(num_blocks):
        # Decide on stride (allow stride=2 only in first few blocks for downsampling)
        stride = 1
        if i < 3 and random.random() < 0.3:
            stride = 2

        # Decide output channels (can increase at stride=2 points)
        if stride == 2:
            out_channels = min(current_channels * 2, base_channels * 4)
        else:
            out_channels = current_channels
            # Small chance to increase channels even without stride
            if random.random() < 0.2:
                out_channels = min(current_channels * 2, base_channels * 4)

        block: BlockConfig = {
            "kernel_size": random.choice(KERNEL_SIZES),
            "expansion": random.choice(EXPANSIONS),
            "use_se": random.random() < 0.5,
            "stride": stride,
            "out_channels": out_channels,
        }
        blocks.append(block)
        current_channels = out_channels

    return {
        "num_blocks": num_blocks,
        "base_channels": base_channels,
        "blocks": blocks,
    }


def config_to_arch_encoding(config: ArchConfig) -> np.ndarray:
    """Convert architecture config to fixed-length vector encoding.

    Encoding format (for max 6 blocks):
    - [0]: num_blocks (normalized)
    - [1]: base_channels (one-hot index)
    - [2:2+6*5]: per-block encodings (kernel_size, expansion, use_se, stride, out_channels)
    """
    max_blocks = 6
    features_per_block = 5

    encoding = np.zeros(2 + max_blocks * features_per_block, dtype=np.float32)

    # Global features
    encoding[0] = config["num_blocks"] / 6.0
    encoding[1] = BASE_CHANNELS.index(config["base_channels"]) / 2.0

    # Per-block features
    for i, block in enumerate(config["blocks"]):
        offset = 2 + i * features_per_block
        encoding[offset + 0] = KERNEL_SIZES.index(block["kernel_size"])
        encoding[offset + 1] = EXPANSIONS.index(block["expansion"]) / 2.0
        encoding[offset + 2] = float(block["use_se"])
        encoding[offset + 3] = float(block["stride"] - 1)
        encoding[offset + 4] = block["out_channels"] / 128.0  # Normalize

    return encoding


def mutate_config(config: ArchConfig, mutation_prob: float = 0.3) -> ArchConfig:
    """Mutate an architecture config with given probability per parameter."""
    config = copy.deepcopy(config)

    # Mutate num_blocks
    if random.random() < mutation_prob:
        delta = random.choice([-1, 1])
        new_num = max(2, min(6, config["num_blocks"] + delta))
        if new_num > config["num_blocks"]:
            # Add a block
            last_ch = config["blocks"][-1]["out_channels"]
            config["blocks"].append({
                "kernel_size": random.choice(KERNEL_SIZES),
                "expansion": random.choice(EXPANSIONS),
                "use_se": random.random() < 0.5,
                "stride": 1,
                "out_channels": last_ch,
            })
        elif new_num < config["num_blocks"]:
            # Remove last block
            config["blocks"].pop()
        config["num_blocks"] = new_num

    # Mutate base_channels
    if random.random() < mutation_prob:
        old_base = config["base_channels"]
        new_base = random.choice(BASE_CHANNELS)
        if new_base != old_base:
            ratio = new_base / old_base
            config["base_channels"] = new_base
            # Scale all channel values
            for block in config["blocks"]:
                block["out_channels"] = int(block["out_channels"] * ratio)
                # Clamp to valid range
                block["out_channels"] = max(new_base, min(new_base * 4, block["out_channels"]))

    # Mutate individual blocks
    for block in config["blocks"]:
        if random.random() < mutation_prob:
            block["kernel_size"] = random.choice(KERNEL_SIZES)
        if random.random() < mutation_prob:
            block["expansion"] = random.choice(EXPANSIONS)
        if random.random() < mutation_prob:
            block["use_se"] = not block["use_se"]
        # Don't mutate stride to maintain spatial dimensions consistency

    return config


def crossover_configs(parent1: ArchConfig, parent2: ArchConfig) -> ArchConfig:
    """Create offspring by crossing over two parent configurations."""
    # Choose num_blocks and base_channels from either parent
    num_blocks = random.choice([parent1["num_blocks"], parent2["num_blocks"]])
    base_channels = random.choice([parent1["base_channels"], parent2["base_channels"]])

    blocks: list[BlockConfig] = []
    current_channels = base_channels

    for i in range(num_blocks):
        # Pick block config from either parent if available
        candidates = []
        if i < len(parent1["blocks"]):
            candidates.append(parent1["blocks"][i])
        if i < len(parent2["blocks"]):
            candidates.append(parent2["blocks"][i])

        if candidates:
            template = random.choice(candidates)
            # Copy and adjust channels to be consistent
            block: BlockConfig = {
                "kernel_size": template["kernel_size"],
                "expansion": template["expansion"],
                "use_se": template["use_se"],
                "stride": template["stride"],
                "out_channels": template["out_channels"],
            }
        else:
            # No candidate, create random block
            block = {
                "kernel_size": random.choice(KERNEL_SIZES),
                "expansion": random.choice(EXPANSIONS),
                "use_se": random.random() < 0.5,
                "stride": 1,
                "out_channels": current_channels,
            }

        # Ensure channel consistency
        if block["stride"] == 2:
            block["out_channels"] = min(current_channels * 2, base_channels * 4)
        else:
            # Keep channels in valid range
            block["out_channels"] = max(base_channels, min(base_channels * 4, block["out_channels"]))

        blocks.append(block)
        current_channels = block["out_channels"]

    return {
        "num_blocks": num_blocks,
        "base_channels": base_channels,
        "blocks": blocks,
    }
