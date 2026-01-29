"""Model tests."""
import torch

from src.models import TinyConvNet


def test_tiny_forward():
    model = TinyConvNet(num_classes=10, input_channels=3)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 10)
