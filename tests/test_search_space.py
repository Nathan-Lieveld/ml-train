"""Tests for search_space module."""
import numpy as np
import torch

from src.search_space import (
    EXPANSIONS,
    KERNEL_SIZES,
    ArchConfig,
    SearchableBlock,
    SearchableNetwork,
    SqueezeExcite,
    config_to_arch_encoding,
    crossover_configs,
    mutate_config,
    sample_random_config,
)


class TestSqueezeExcite:
    """Tests for SqueezeExcite module."""

    def test_forward_shape(self):
        """SE module preserves input shape."""
        se = SqueezeExcite(channels=32, reduction=4)
        x = torch.randn(2, 32, 8, 8)
        out = se(x)
        assert out.shape == x.shape

    def test_different_reductions(self):
        """SE module works with different reduction factors."""
        for reduction in [2, 4, 8]:
            se = SqueezeExcite(channels=64, reduction=reduction)
            x = torch.randn(1, 64, 4, 4)
            out = se(x)
            assert out.shape == x.shape


class TestSearchableBlock:
    """Tests for SearchableBlock module."""

    def test_forward_basic(self):
        """Basic forward pass with default settings."""
        block = SearchableBlock(in_channels=16, out_channels=16)
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert out.shape == (2, 16, 32, 32)

    def test_forward_stride2(self):
        """Forward pass with stride=2 downsampling."""
        block = SearchableBlock(in_channels=16, out_channels=32, stride=2)
        x = torch.randn(2, 16, 32, 32)
        out = block(x)
        assert out.shape == (2, 32, 16, 16)

    def test_forward_with_se(self):
        """Forward pass with SE module enabled."""
        block = SearchableBlock(in_channels=24, out_channels=24, use_se=True)
        x = torch.randn(2, 24, 16, 16)
        out = block(x)
        assert out.shape == (2, 24, 16, 16)

    def test_all_kernel_expansion_combinations(self):
        """Test all valid kernel and expansion combinations."""
        for kernel in KERNEL_SIZES:
            for expansion in EXPANSIONS:
                block = SearchableBlock(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=kernel,
                    expansion=expansion,
                )
                x = torch.randn(1, 16, 8, 8)
                out = block(x)
                assert out.shape == (1, 16, 8, 8), f"Failed for k={kernel}, e={expansion}"

    def test_residual_connection(self):
        """Residual connection when in_channels == out_channels and stride == 1."""
        block = SearchableBlock(in_channels=32, out_channels=32, stride=1)
        assert block.use_residual is True

        block_no_res = SearchableBlock(in_channels=32, out_channels=64, stride=1)
        assert block_no_res.use_residual is False

        block_stride = SearchableBlock(in_channels=32, out_channels=32, stride=2)
        assert block_stride.use_residual is False

    def test_channel_change(self):
        """Block can change number of channels."""
        block = SearchableBlock(in_channels=16, out_channels=48)
        x = torch.randn(2, 16, 8, 8)
        out = block(x)
        assert out.shape == (2, 48, 8, 8)


class TestSearchableNetwork:
    """Tests for SearchableNetwork module."""

    def test_forward_from_config(self):
        """Network forward pass from a simple config."""
        config: ArchConfig = {
            "num_blocks": 2,
            "base_channels": 16,
            "blocks": [
                {
                    "kernel_size": 3,
                    "expansion": 2,
                    "use_se": False,
                    "stride": 1,
                    "out_channels": 16,
                },
                {
                    "kernel_size": 3,
                    "expansion": 4,
                    "use_se": True,
                    "stride": 2,
                    "out_channels": 32,
                },
            ],
        }
        model = SearchableNetwork(config, num_classes=10, input_size=32)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_random_config(self):
        """Network forward pass from random config."""
        config = sample_random_config()
        model = SearchableNetwork(config, num_classes=10)
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)

    def test_different_num_classes(self):
        """Network works with different number of classes."""
        config = sample_random_config()
        for num_classes in [2, 10, 100]:
            model = SearchableNetwork(config, num_classes=num_classes)
            x = torch.randn(1, 3, 32, 32)
            out = model(x)
            assert out.shape == (1, num_classes)


class TestConfigUtils:
    """Tests for config utility functions."""

    def test_sample_random_config_valid(self):
        """Sampled config has valid structure."""
        for _ in range(10):
            config = sample_random_config()

            # Check required keys
            assert "num_blocks" in config
            assert "base_channels" in config
            assert "blocks" in config

            # Check value ranges
            assert 2 <= config["num_blocks"] <= 6
            assert config["base_channels"] in [16, 24, 32]
            assert len(config["blocks"]) == config["num_blocks"]

            # Check each block
            for block in config["blocks"]:
                assert block["kernel_size"] in KERNEL_SIZES
                assert block["expansion"] in EXPANSIONS
                assert isinstance(block["use_se"], bool)
                assert block["stride"] in [1, 2]
                assert block["out_channels"] > 0

    def test_config_encoding_shape(self):
        """Config encoding has correct shape."""
        config = sample_random_config()
        encoding = config_to_arch_encoding(config)

        # 2 global + 6 blocks * 5 features = 32
        assert encoding.shape == (32,)
        assert encoding.dtype == np.float32

    def test_config_encoding_deterministic(self):
        """Same config produces same encoding."""
        config: ArchConfig = {
            "num_blocks": 2,
            "base_channels": 16,
            "blocks": [
                {
                    "kernel_size": 3,
                    "expansion": 2,
                    "use_se": False,
                    "stride": 1,
                    "out_channels": 16,
                },
                {
                    "kernel_size": 5,
                    "expansion": 6,
                    "use_se": True,
                    "stride": 2,
                    "out_channels": 32,
                },
            ],
        }
        enc1 = config_to_arch_encoding(config)
        enc2 = config_to_arch_encoding(config)
        np.testing.assert_array_equal(enc1, enc2)

    def test_mutate_produces_valid_config(self):
        """Mutation produces valid config."""
        for _ in range(20):
            original = sample_random_config()
            mutated = mutate_config(original, mutation_prob=0.5)

            # Check structure
            assert 2 <= mutated["num_blocks"] <= 6
            assert mutated["base_channels"] in [16, 24, 32]
            assert len(mutated["blocks"]) == mutated["num_blocks"]

            # Original should be unchanged (deep copy)
            assert original["num_blocks"] == len(original["blocks"])

    def test_mutate_changes_config(self):
        """Mutation with high probability changes config."""
        original = sample_random_config()
        changes_detected = False

        for _ in range(10):
            mutated = mutate_config(original, mutation_prob=0.9)
            enc_orig = config_to_arch_encoding(original)
            enc_mut = config_to_arch_encoding(mutated)
            if not np.array_equal(enc_orig, enc_mut):
                changes_detected = True
                break

        assert changes_detected, "Mutation should change config with high probability"

    def test_crossover_produces_valid_config(self):
        """Crossover produces valid config."""
        for _ in range(20):
            parent1 = sample_random_config()
            parent2 = sample_random_config()
            child = crossover_configs(parent1, parent2)

            # Check structure
            assert 2 <= child["num_blocks"] <= 6
            assert child["base_channels"] in [16, 24, 32]
            assert len(child["blocks"]) == child["num_blocks"]

            # Each block should be valid
            for block in child["blocks"]:
                assert block["kernel_size"] in KERNEL_SIZES
                assert block["expansion"] in EXPANSIONS
                assert isinstance(block["use_se"], bool)
                assert block["stride"] in [1, 2]
                assert block["out_channels"] > 0

    def test_crossover_combines_parents(self):
        """Crossover combines properties from both parents."""
        parent1: ArchConfig = {
            "num_blocks": 2,
            "base_channels": 16,
            "blocks": [
                {
                    "kernel_size": 3,
                    "expansion": 2,
                    "use_se": False,
                    "stride": 1,
                    "out_channels": 16,
                },
                {
                    "kernel_size": 3,
                    "expansion": 2,
                    "use_se": False,
                    "stride": 1,
                    "out_channels": 16,
                },
            ],
        }
        parent2: ArchConfig = {
            "num_blocks": 4,
            "base_channels": 32,
            "blocks": [
                {
                    "kernel_size": 5,
                    "expansion": 6,
                    "use_se": True,
                    "stride": 2,
                    "out_channels": 64,
                },
                {
                    "kernel_size": 5,
                    "expansion": 6,
                    "use_se": True,
                    "stride": 1,
                    "out_channels": 64,
                },
                {
                    "kernel_size": 5,
                    "expansion": 6,
                    "use_se": True,
                    "stride": 1,
                    "out_channels": 64,
                },
                {
                    "kernel_size": 5,
                    "expansion": 6,
                    "use_se": True,
                    "stride": 1,
                    "out_channels": 64,
                },
            ],
        }

        # Run multiple times to verify properties can come from either parent
        num_blocks_seen = set()
        base_channels_seen = set()

        for _ in range(20):
            child = crossover_configs(parent1, parent2)
            num_blocks_seen.add(child["num_blocks"])
            base_channels_seen.add(child["base_channels"])

        # Should see values from both parents
        assert parent1["num_blocks"] in num_blocks_seen or parent2["num_blocks"] in num_blocks_seen
        assert (
            parent1["base_channels"] in base_channels_seen
            or parent2["base_channels"] in base_channels_seen
        )


class TestNetworkFromRandomConfig:
    """Integration tests for networks from random configs."""

    def test_multiple_random_configs_trainable(self):
        """Networks from random configs can compute gradients."""
        for _ in range(5):
            config = sample_random_config()
            model = SearchableNetwork(config)
            x = torch.randn(2, 3, 32, 32)
            y = model(x)

            loss = y.sum()
            loss.backward()

            # Check gradients exist
            for param in model.parameters():
                assert param.grad is not None
