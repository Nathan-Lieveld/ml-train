"""Tests for CoreML export functionality."""
import tempfile
from pathlib import Path

import pytest

# Skip all tests if coremltools is not available
coremltools = pytest.importorskip("coremltools")

from src.coreml_export import (  # noqa: E402
    export_coreml,
    export_coreml_flexible,
    validate_coreml_model,
)
from src.models import TinyConvNet  # noqa: E402
from src.search_space import SearchableNetwork, sample_random_config  # noqa: E402


class TestExportTinyConvNet:
    """Tests for exporting TinyConvNet to CoreML."""

    def test_export_tinyconvnet(self):
        """Basic export of TinyConvNet."""
        model = TinyConvNet(num_classes=10)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.mlpackage"
            export_coreml(model, str(output_path))

            assert output_path.exists()
            assert output_path.is_dir()  # mlpackage is a directory

    def test_export_with_custom_input_shape(self):
        """Export with non-default input shape."""
        model = TinyConvNet()
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.mlpackage"
            export_coreml(model, str(output_path), input_shape=(1, 3, 64, 64))

            assert output_path.exists()


class TestExportSearchableNetwork:
    """Tests for exporting SearchableNetwork to CoreML."""

    def test_export_searchable_network(self):
        """Export a SearchableNetwork to CoreML."""
        config = sample_random_config()
        model = SearchableNetwork(config, num_classes=10)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "searchable.mlpackage"
            export_coreml(model, str(output_path))

            assert output_path.exists()

    def test_export_multiple_random_configs(self):
        """Export multiple random configurations."""
        for _ in range(3):
            config = sample_random_config()
            model = SearchableNetwork(config)
            model.eval()

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "model.mlpackage"
                export_coreml(model, str(output_path))
                assert output_path.exists()


class TestExportPrecision:
    """Tests for export precision options."""

    def test_export_fp16_precision(self):
        """Verify FP16 precision export."""
        model = TinyConvNet()
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_fp16.mlpackage"
            export_coreml(model, str(output_path))

            specs = validate_coreml_model(str(output_path))
            assert specs["precision"] == "FP16"

    def test_export_quantized(self):
        """Export with INT8 quantization."""
        model = TinyConvNet()
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_quant.mlpackage"
            export_coreml(model, str(output_path), quantize=True)

            assert output_path.exists()


class TestExportFlexible:
    """Tests for flexible input shape export."""

    def test_export_flexible_shapes(self):
        """Export with multiple enumerated shapes."""
        model = TinyConvNet()
        model.eval()

        shapes = [
            (1, 3, 32, 32),
            (1, 3, 64, 64),
            (4, 3, 32, 32),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_flex.mlpackage"
            export_coreml_flexible(model, str(output_path), shapes=shapes)

            assert output_path.exists()


class TestValidation:
    """Tests for CoreML model validation."""

    def test_validate_model_specs(self):
        """Validate exported model returns correct specs."""
        model = TinyConvNet(num_classes=10)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.mlpackage"
            export_coreml(model, str(output_path))

            specs = validate_coreml_model(str(output_path))

            assert "input_specs" in specs
            assert "output_specs" in specs
            assert "precision" in specs
            assert len(specs["input_specs"]) == 1
            assert specs["input_specs"][0]["name"] == "input"


class TestComputeUnits:
    """Tests for different compute unit configurations."""

    @pytest.mark.parametrize(
        "compute_units",
        ["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
    )
    def test_export_with_compute_units(self, compute_units):
        """Export with different compute unit settings."""
        model = TinyConvNet()
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / f"model_{compute_units}.mlpackage"
            export_coreml(model, str(output_path), compute_units=compute_units)

            assert output_path.exists()
