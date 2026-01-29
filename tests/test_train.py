"""Training tests."""
import torch
import torch.nn as nn

from src.models import TinyConvNet
from src.train import get_dataloaders, train_epoch, validate


def test_train_epoch_single_batch():
    """Verify training step runs without error."""
    model = TinyConvNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    # Create a small fake loader
    inputs = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))
    loader = [(inputs, targets)]

    loss, acc = train_epoch(model, loader, optimizer, criterion, device)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_validate_returns_metrics():
    """Verify validation returns loss and accuracy."""
    model = TinyConvNet()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    inputs = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))
    loader = [(inputs, targets)]

    loss, acc = validate(model, loader, criterion, device)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_get_dataloaders():
    """Verify CIFAR-10 loaders work."""
    train_loader, val_loader = get_dataloaders(batch_size=4)

    # Check we can iterate
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert train_batch[0].shape == (4, 3, 32, 32)
    assert train_batch[1].shape == (4,)
    assert val_batch[0].shape == (4, 3, 32, 32)
    assert val_batch[1].shape == (4,)
