"""Model tests."""
import torch

from src.models import TinyConvNet


def test_tiny_forward():
    model = TinyConvNet(num_classes=10, input_channels=3)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 10)


def test_tiny_forward_batch():
    """Test batch size > 1."""
    model = TinyConvNet()
    x = torch.randn(16, 3, 32, 32)
    out = model(x)
    assert out.shape == (16, 10)


def test_tiny_param_count():
    """Verify < 100K params."""
    model = TinyConvNet()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 100_000


def test_tiny_custom_classes():
    """Test custom num_classes and input_channels."""
    model = TinyConvNet(num_classes=5, input_channels=1)
    x = torch.randn(4, 1, 32, 32)
    out = model(x)
    assert out.shape == (4, 5)
