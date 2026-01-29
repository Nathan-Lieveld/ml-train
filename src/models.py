"""Model definitions for edge deployment."""
import torch
import torch.nn as nn


class TinyConvNet(nn.Module):
    """Example tiny CNN (~50K params, ~50KB quantized)."""

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
