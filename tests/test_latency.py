"""Tests for latency module."""
import json
import tempfile
from pathlib import Path

import pytest

from src.latency import (
    LatencyTable,
    OpConfig,
    create_minimal_model,
    estimate_network_latency,
    generate_lut_configs,
)
from src.search_space import ArchConfig


class TestOpConfig:
    """Tests for OpConfig dataclass."""

    def test_to_key_deterministic(self):
        """Same config produces same key."""
        config1 = OpConfig(
            op_type="block",
            channels=32,
            kernel_size=3,
            expansion=4,
            use_se=True,
            stride=1,
            input_size=16,
        )
        config2 = OpConfig(
            op_type="block",
            channels=32,
            kernel_size=3,
            expansion=4,
            use_se=True,
            stride=1,
            input_size=16,
        )
        assert config1.to_key() == config2.to_key()

    def test_to_key_different_configs(self):
        """Different configs produce different keys."""
        config1 = OpConfig(op_type="block", channels=32, kernel_size=3)
        config2 = OpConfig(op_type="block", channels=64, kernel_size=3)
        assert config1.to_key() != config2.to_key()


class TestLatencyTable:
    """Tests for LatencyTable class."""

    def test_add_and_lookup(self):
        """Add entry and look it up."""
        table = LatencyTable()
        config = OpConfig(op_type="block", channels=32, kernel_size=3, expansion=4)

        table.add(config, mean_ms=1.5, std_ms=0.1, samples=100)

        entry = table.lookup(config)
        assert entry is not None
        assert entry.mean_ms == 1.5
        assert entry.std_ms == 0.1
        assert entry.samples == 100

    def test_lookup_missing(self):
        """Lookup returns None for missing config."""
        table = LatencyTable()
        config = OpConfig(op_type="block", channels=32)

        assert table.lookup(config) is None

    def test_contains(self):
        """Test __contains__ method."""
        table = LatencyTable()
        config = OpConfig(op_type="conv", channels=16)

        assert config not in table
        table.add(config, mean_ms=0.5, std_ms=0.05, samples=50)
        assert config in table

    def test_len(self):
        """Test __len__ method."""
        table = LatencyTable()
        assert len(table) == 0

        table.add(OpConfig(op_type="conv", channels=16), 0.5, 0.05, 50)
        assert len(table) == 1

        table.add(OpConfig(op_type="conv", channels=32), 0.8, 0.08, 50)
        assert len(table) == 2

    def test_save_and_load(self):
        """Save and load table from JSON."""
        table = LatencyTable()
        configs = [
            OpConfig(op_type="block", channels=16, kernel_size=3, expansion=2, use_se=False),
            OpConfig(op_type="block", channels=32, kernel_size=5, expansion=4, use_se=True),
            OpConfig(op_type="stem", channels=16, input_size=32),
        ]

        for i, config in enumerate(configs):
            table.add(config, mean_ms=float(i + 1), std_ms=0.1 * (i + 1), samples=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lut.json"
            table.save(path)

            # Verify file exists and is valid JSON
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 3

            # Load and verify
            loaded = LatencyTable.load(path)
            assert len(loaded) == 3

            for config in configs:
                orig = table.lookup(config)
                load = loaded.lookup(config)
                assert load is not None
                assert load.mean_ms == orig.mean_ms
                assert load.std_ms == orig.std_ms
                assert load.samples == orig.samples

    def test_overwrite_entry(self):
        """Adding same config overwrites existing entry."""
        table = LatencyTable()
        config = OpConfig(op_type="conv", channels=32)

        table.add(config, mean_ms=1.0, std_ms=0.1, samples=50)
        table.add(config, mean_ms=2.0, std_ms=0.2, samples=100)

        entry = table.lookup(config)
        assert entry.mean_ms == 2.0
        assert len(table) == 1


class TestCreateMinimalModel:
    """Tests for create_minimal_model function."""

    def test_create_block_model(self):
        """Create minimal model for block op."""
        import torch

        config = OpConfig(op_type="block", channels=32, kernel_size=3, expansion=4, input_size=16)
        model = create_minimal_model(config)

        x = torch.randn(1, 32, 16, 16)
        y = model(x)
        assert y.shape[0] == 1

    def test_create_stem_model(self):
        """Create minimal model for stem op."""
        import torch

        config = OpConfig(op_type="stem", channels=16, input_size=32)
        model = create_minimal_model(config)

        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        assert y.shape == (1, 16, 32, 32)

    def test_create_classifier_model(self):
        """Create minimal model for classifier op."""
        import torch

        config = OpConfig(op_type="classifier", channels=64, input_size=1)
        model = create_minimal_model(config)

        x = torch.randn(1, 64)
        y = model(x)
        assert y.shape == (1, 10)

    def test_create_se_model(self):
        """Create minimal model for SE op."""
        import torch

        config = OpConfig(op_type="se", channels=32, input_size=8)
        model = create_minimal_model(config)

        x = torch.randn(1, 32, 8, 8)
        y = model(x)
        assert y.shape == x.shape


class TestEstimateNetworkLatency:
    """Tests for estimate_network_latency function."""

    def test_estimate_with_mock_lut(self):
        """Estimate latency with mock LUT data."""
        # Create a simple architecture
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
                    "use_se": False,
                    "stride": 2,
                    "out_channels": 32,
                },
            ],
        }

        # Create mock LUT
        lut = LatencyTable()
        lut.add(OpConfig(op_type="stem", channels=16, input_size=32), mean_ms=0.5, std_ms=0.05, samples=100)
        lut.add(
            OpConfig(op_type="block", channels=16, kernel_size=3, expansion=2, use_se=False, stride=1, input_size=32),
            mean_ms=1.0,
            std_ms=0.1,
            samples=100,
        )
        lut.add(
            OpConfig(op_type="block", channels=16, kernel_size=3, expansion=4, use_se=False, stride=2, input_size=32),
            mean_ms=1.5,
            std_ms=0.15,
            samples=100,
        )
        lut.add(OpConfig(op_type="classifier", channels=32, input_size=1), mean_ms=0.2, std_ms=0.02, samples=100)

        latency = estimate_network_latency(config, lut, input_size=32)

        # Should sum: stem(0.5) + block1(1.0) + block2(1.5) + classifier(0.2) = 3.2
        assert latency == pytest.approx(3.2, rel=0.01)

    def test_estimate_with_missing_entries(self):
        """Estimate handles missing LUT entries gracefully."""
        config: ArchConfig = {
            "num_blocks": 1,
            "base_channels": 16,
            "blocks": [
                {
                    "kernel_size": 3,
                    "expansion": 2,
                    "use_se": False,
                    "stride": 1,
                    "out_channels": 16,
                },
            ],
        }

        # Empty LUT
        lut = LatencyTable()
        latency = estimate_network_latency(config, lut)

        # Should return 0 when no entries found
        assert latency == 0.0


class TestGenerateLutConfigs:
    """Tests for generate_lut_configs function."""

    def test_generates_configs(self):
        """Generate produces valid configs."""
        configs = generate_lut_configs()
        assert len(configs) > 0

        # Check that we have different op types
        op_types = {c.op_type for c in configs}
        assert "stem" in op_types
        assert "block" in op_types
        assert "classifier" in op_types

    def test_covers_kernel_sizes(self):
        """Generated configs cover all kernel sizes."""
        configs = generate_lut_configs()
        kernel_sizes = {c.kernel_size for c in configs if c.op_type == "block"}
        assert 3 in kernel_sizes
        assert 5 in kernel_sizes

    def test_covers_expansions(self):
        """Generated configs cover all expansion ratios."""
        configs = generate_lut_configs()
        expansions = {c.expansion for c in configs if c.op_type == "block"}
        assert 2 in expansions
        assert 4 in expansions
        assert 6 in expansions
