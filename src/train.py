"""Training script."""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from .models import TinyConvNet


def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TinyConvNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # TODO: Add data loading
    # TODO: Add training loop

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
