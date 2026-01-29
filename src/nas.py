"""Neural Architecture Search with evolutionary algorithm."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .latency import LatencyTable, estimate_network_latency
from .search_space import (
    ArchConfig,
    SearchableNetwork,
    crossover_configs,
    mutate_config,
    sample_random_config,
)
from .train import train_epoch, validate


def get_subset_dataloaders(
    batch_size: int,
    subset_fraction: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and validation dataloaders with subset sampling.

    Args:
        batch_size: Batch size for dataloaders
        subset_fraction: Fraction of training data to use (0-1)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=val_transform
    )

    # Create subset of training data
    num_train = len(train_dataset)
    num_subset = int(num_train * subset_fraction)
    indices = random.sample(range(num_train), num_subset)
    train_subset = Subset(train_dataset, indices)

    # Use smaller subset for validation too
    num_val = len(val_dataset)
    num_val_subset = int(num_val * subset_fraction)
    val_indices = random.sample(range(num_val), num_val_subset)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader


def train_and_eval(
    config: ArchConfig,
    epochs: int = 5,
    subset_fraction: float = 0.1,
    batch_size: int = 64,
    lr: float = 0.001,
    device: Optional[torch.device] = None,
) -> float:
    """Train a model from config and return validation accuracy.

    This is the proxy training function used during NAS to quickly
    evaluate candidate architectures.

    Args:
        config: Architecture configuration
        epochs: Number of training epochs
        subset_fraction: Fraction of data to use
        batch_size: Training batch size
        lr: Learning rate
        device: Device to train on (auto-detected if None)

    Returns:
        Validation accuracy (0-1)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SearchableNetwork(config, num_classes=10)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_subset_dataloaders(batch_size, subset_fraction)

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)

    _, val_acc = validate(model, val_loader, criterion, device)
    return val_acc


def _update_pareto(
    frontier: list[tuple[ArchConfig, float, float]],
    config: ArchConfig,
    accuracy: float,
    latency: float,
) -> list[tuple[ArchConfig, float, float]]:
    """Update Pareto frontier with a new candidate.

    A point is Pareto-optimal if no other point dominates it
    (i.e., has both better accuracy AND lower latency).

    Args:
        frontier: Current Pareto frontier
        config: New architecture config
        accuracy: Accuracy of new config
        latency: Latency of new config

    Returns:
        Updated Pareto frontier
    """
    # Check if new point is dominated by any existing point
    for _, acc, lat in frontier:
        if acc >= accuracy and lat <= latency:
            # New point is dominated, don't add it
            if acc > accuracy or lat < latency:
                return frontier

    # New point is not dominated, add it and remove dominated points
    new_frontier = []
    for item in frontier:
        _, acc, lat = item
        # Keep point if it's not dominated by the new point
        if not (accuracy >= acc and latency <= lat and (accuracy > acc or latency < lat)):
            new_frontier.append(item)

    new_frontier.append((config, accuracy, latency))
    return new_frontier


def evolutionary_search(
    generations: int,
    pop_size: int,
    lut: LatencyTable,
    target_latency_ms: float,
    lambda_penalty: float = 1.0,
    accuracy_fn: Optional[Callable[[ArchConfig], float]] = None,
    save_dir: Optional[str] = None,
    epochs_per_eval: int = 5,
    subset_fraction: float = 0.1,
) -> list[tuple[ArchConfig, float, float]]:
    """Run evolutionary architecture search.

    Uses a simple (mu + lambda) evolutionary strategy:
    1. Initialize random population
    2. Evaluate fitness (accuracy - penalty * max(0, latency - target))
    3. Select top 50% as parents
    4. Generate offspring via crossover + mutation
    5. Repeat for specified generations

    Args:
        generations: Number of generations to evolve
        pop_size: Population size
        lut: Latency lookup table
        target_latency_ms: Target latency constraint
        lambda_penalty: Penalty weight for exceeding target latency
        accuracy_fn: Function to evaluate accuracy (defaults to train_and_eval)
        save_dir: Directory to save results
        epochs_per_eval: Training epochs for proxy evaluation
        subset_fraction: Fraction of data for proxy training

    Returns:
        Pareto frontier as list of (config, accuracy, latency) tuples
    """
    from tqdm import tqdm

    if accuracy_fn is None:
        def accuracy_fn(cfg: ArchConfig) -> float:
            return train_and_eval(cfg, epochs=epochs_per_eval, subset_fraction=subset_fraction)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    # Initialize population
    population: list[tuple[ArchConfig, float, float]] = []
    print(f"Initializing population of {pop_size}...")

    for _ in tqdm(range(pop_size), desc="Init population"):
        config = sample_random_config()
        accuracy = accuracy_fn(config)
        latency = estimate_network_latency(config, lut)
        population.append((config, accuracy, latency))

    # Track Pareto frontier
    pareto_frontier: list[tuple[ArchConfig, float, float]] = []
    for config, accuracy, latency in population:
        pareto_frontier = _update_pareto(pareto_frontier, config, accuracy, latency)

    def fitness(accuracy: float, latency: float) -> float:
        penalty = max(0, latency - target_latency_ms)
        return accuracy - lambda_penalty * penalty

    # Evolution loop
    for gen in range(generations):
        print(f"\n=== Generation {gen + 1}/{generations} ===")

        # Sort by fitness
        population.sort(key=lambda x: fitness(x[1], x[2]), reverse=True)

        # Report stats
        best = population[0]
        print(f"Best: acc={best[1]:.4f}, lat={best[2]:.2f}ms, fitness={fitness(best[1], best[2]):.4f}")
        print(f"Pareto frontier size: {len(pareto_frontier)}")

        # Select top 50% as parents
        num_parents = pop_size // 2
        parents = population[:num_parents]

        # Generate offspring
        offspring: list[tuple[ArchConfig, float, float]] = []
        num_offspring = pop_size - num_parents

        for _ in tqdm(range(num_offspring), desc=f"Gen {gen + 1} offspring"):
            # Select two parents
            p1, p2 = random.sample(parents, 2)

            # Crossover
            child_config = crossover_configs(p1[0], p2[0])

            # Mutation
            child_config = mutate_config(child_config, mutation_prob=0.3)

            # Evaluate
            accuracy = accuracy_fn(child_config)
            latency = estimate_network_latency(child_config, lut)

            offspring.append((child_config, accuracy, latency))

            # Update Pareto frontier
            pareto_frontier = _update_pareto(pareto_frontier, child_config, accuracy, latency)

        # Next generation = parents + offspring
        population = parents + offspring

        # Save checkpoint
        if save_dir:
            checkpoint = {
                "generation": gen + 1,
                "population": [
                    {"config": cfg, "accuracy": acc, "latency": lat}
                    for cfg, acc, lat in population
                ],
                "pareto_frontier": [
                    {"config": cfg, "accuracy": acc, "latency": lat}
                    for cfg, acc, lat in pareto_frontier
                ],
            }
            with open(save_path / "checkpoint.json", "w") as f:
                json.dump(checkpoint, f, indent=2)

    # Save final Pareto frontier
    if save_dir:
        frontier_data = [
            {"config": cfg, "accuracy": acc, "latency": lat}
            for cfg, acc, lat in pareto_frontier
        ]
        with open(save_path / "pareto_frontier.json", "w") as f:
            json.dump(frontier_data, f, indent=2)
        print(f"\nSaved Pareto frontier to {save_path / 'pareto_frontier.json'}")

    return pareto_frontier


def main():
    """CLI entry point for NAS search."""
    parser = argparse.ArgumentParser(description="Run evolutionary NAS")
    parser.add_argument("--lut-path", type=str, required=True, help="Path to latency LUT")
    parser.add_argument("--target-latency", type=float, required=True, help="Target latency in ms")
    parser.add_argument("--generations", type=int, default=20, help="Number of generations")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size")
    parser.add_argument("--lambda-penalty", type=float, default=1.0, help="Latency penalty weight")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per evaluation")
    parser.add_argument("--subset", type=float, default=0.1, help="Data subset fraction")
    parser.add_argument("--save-dir", type=str, default="nas_results", help="Output directory")
    args = parser.parse_args()

    lut = LatencyTable.load(args.lut_path)
    print(f"Loaded LUT with {len(lut)} entries")

    frontier = evolutionary_search(
        generations=args.generations,
        pop_size=args.pop_size,
        lut=lut,
        target_latency_ms=args.target_latency,
        lambda_penalty=args.lambda_penalty,
        save_dir=args.save_dir,
        epochs_per_eval=args.epochs,
        subset_fraction=args.subset,
    )

    print(f"\n=== Final Pareto Frontier ({len(frontier)} architectures) ===")
    for i, (config, acc, lat) in enumerate(sorted(frontier, key=lambda x: x[2])):
        print(f"{i + 1}. acc={acc:.4f}, lat={lat:.2f}ms, blocks={config['num_blocks']}, base_ch={config['base_channels']}")


if __name__ == "__main__":
    main()
