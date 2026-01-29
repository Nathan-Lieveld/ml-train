"""Export tests."""
import tempfile
from pathlib import Path

import pytest
import torch

from src.export import export_onnx, export_torchscript, load_model_from_checkpoint
from src.models import TinyConvNet

try:
    import onnx
    import onnxruntime as ort

    HAS_ONNX_RUNTIME = True
except ImportError:
    HAS_ONNX_RUNTIME = False


@pytest.mark.skipif(not HAS_ONNX_RUNTIME, reason="onnx/onnxruntime not available")
def test_export_onnx():
    """Verify ONNX export and validation."""
    model = TinyConvNet()
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "model.onnx"
        export_onnx(model, str(output_path))

        assert output_path.exists()

        # Validate ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        # Run inference with onnxruntime
        session = ort.InferenceSession(str(output_path))
        dummy_input = torch.randn(2, 3, 32, 32).numpy()
        outputs = session.run(None, {"input": dummy_input})
        assert outputs[0].shape == (2, 10)


def test_export_torchscript():
    """Verify TorchScript export and inference."""
    model = TinyConvNet()
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "model.pt"
        export_torchscript(model, str(output_path))

        assert output_path.exists()

        # Load and run inference
        loaded = torch.jit.load(str(output_path))
        dummy_input = torch.randn(2, 3, 32, 32)
        output = loaded(dummy_input)
        assert output.shape == (2, 10)


def test_load_model_from_checkpoint():
    """Verify checkpoint loading."""
    model = TinyConvNet()
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"
        checkpoint = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "best_acc": 0.85,
        }
        torch.save(checkpoint, checkpoint_path)

        loaded_model = load_model_from_checkpoint(str(checkpoint_path), device)

        assert not loaded_model.training  # Should be in eval mode
        output = loaded_model(torch.randn(1, 3, 32, 32))
        assert output.shape == (1, 10)
