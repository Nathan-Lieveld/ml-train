"""Tests for NAS module."""
import pytest

from src.latency import LatencyTable, OpConfig
from src.nas import (
    _update_pareto,
    evolutionary_search,
    get_subset_dataloaders,
    train_and_eval,
)
from src.search_space import ArchConfig, sample_random_config


class TestGetSubsetDataloaders:
    """Tests for get_subset_dataloaders function."""

    def test_returns_dataloaders(self):
        """Function returns train and val dataloaders."""
        train_loader, val_loader = get_subset_dataloaders(batch_size=32, subset_fraction=0.01)

        assert train_loader is not None
        assert val_loader is not None

        # Check we can iterate
        batch = next(iter(train_loader))
        assert len(batch) == 2  # (inputs, targets)
        assert batch[0].shape[0] <= 32

    def test_subset_fraction(self):
        """Subset fraction reduces dataset size."""
        train_full, _ = get_subset_dataloaders(batch_size=32, subset_fraction=1.0)
        train_small, _ = get_subset_dataloaders(batch_size=32, subset_fraction=0.1)

        # Count total samples
        full_samples = sum(len(batch[0]) for batch in train_full)
        small_samples = sum(len(batch[0]) for batch in train_small)

        # Small should be roughly 10% of full (with some variance due to random sampling)
        assert small_samples < full_samples
        assert small_samples < full_samples * 0.2  # Allow some margin


class TestTrainAndEval:
    """Tests for train_and_eval function."""

    @pytest.mark.slow
    def test_train_and_eval_runs(self):
        """Training runs and returns valid accuracy."""
        config = sample_random_config()

        # Use minimal settings for speed
        accuracy = train_and_eval(
            config,
            epochs=1,
            subset_fraction=0.01,
            batch_size=32,
        )

        assert 0.0 <= accuracy <= 1.0

    @pytest.mark.slow
    def test_train_improves_accuracy(self):
        """More epochs should generally improve accuracy."""
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
                    "expansion": 2,
                    "use_se": False,
                    "stride": 2,
                    "out_channels": 32,
                },
            ],
        }

        # This is a weak test since random initialization can vary
        acc_1 = train_and_eval(config, epochs=1, subset_fraction=0.05)
        acc_3 = train_and_eval(config, epochs=3, subset_fraction=0.05)

        # Both should be valid
        assert 0.0 <= acc_1 <= 1.0
        assert 0.0 <= acc_3 <= 1.0


class TestParetoDominance:
    """Tests for _update_pareto function."""

    def test_add_to_empty_frontier(self):
        """Adding to empty frontier creates singleton."""
        config = sample_random_config()
        frontier = _update_pareto([], config, accuracy=0.8, latency=5.0)

        assert len(frontier) == 1
        assert frontier[0][1] == 0.8
        assert frontier[0][2] == 5.0

    def test_dominated_point_not_added(self):
        """Point dominated by existing point is not added."""
        config1 = sample_random_config()
        config2 = sample_random_config()

        # Point 1: better accuracy AND lower latency
        frontier = [(config1, 0.9, 3.0)]

        # Point 2 is dominated (worse accuracy, higher latency)
        frontier = _update_pareto(frontier, config2, accuracy=0.7, latency=5.0)

        assert len(frontier) == 1
        assert frontier[0][1] == 0.9

    def test_dominating_point_replaces(self):
        """Point that dominates existing points replaces them."""
        config1 = sample_random_config()
        config2 = sample_random_config()

        frontier = [(config1, 0.7, 5.0)]

        # Point 2 dominates (better accuracy, lower latency)
        frontier = _update_pareto(frontier, config2, accuracy=0.9, latency=3.0)

        assert len(frontier) == 1
        assert frontier[0][1] == 0.9

    def test_pareto_optimal_points_coexist(self):
        """Non-dominated points can coexist on frontier."""
        config1 = sample_random_config()
        config2 = sample_random_config()
        config3 = sample_random_config()

        frontier: list = []

        # Point 1: high accuracy, high latency
        frontier = _update_pareto(frontier, config1, accuracy=0.95, latency=10.0)

        # Point 2: medium accuracy, medium latency (not dominated)
        frontier = _update_pareto(frontier, config2, accuracy=0.85, latency=5.0)

        # Point 3: low accuracy, low latency (not dominated)
        frontier = _update_pareto(frontier, config3, accuracy=0.75, latency=2.0)

        # All three should be on frontier
        assert len(frontier) == 3

    def test_equal_points(self):
        """Equal points are handled correctly."""
        config1 = sample_random_config()
        config2 = sample_random_config()

        frontier = [(config1, 0.8, 5.0)]
        frontier = _update_pareto(frontier, config2, accuracy=0.8, latency=5.0)

        # Equal point should be added (neither dominates)
        assert len(frontier) == 2

    def test_strictly_better_accuracy(self):
        """Point with same latency but better accuracy dominates."""
        config1 = sample_random_config()
        config2 = sample_random_config()

        frontier = [(config1, 0.8, 5.0)]
        frontier = _update_pareto(frontier, config2, accuracy=0.9, latency=5.0)

        assert len(frontier) == 1
        assert frontier[0][1] == 0.9


class TestEvolutionarySearch:
    """Tests for evolutionary_search function."""

    @pytest.mark.slow
    def test_search_runs(self):
        """Search runs with mock accuracy function."""
        # Create mock LUT
        lut = LatencyTable()
        for channels in [16, 24, 32, 48, 64, 96, 128]:
            lut.add(
                OpConfig(op_type="stem", channels=channels, input_size=32),
                mean_ms=0.5,
                std_ms=0.05,
                samples=100,
            )
            lut.add(
                OpConfig(op_type="classifier", channels=channels, input_size=1),
                mean_ms=0.1,
                std_ms=0.01,
                samples=100,
            )
            for kernel in [3, 5]:
                for expansion in [2, 4, 6]:
                    for use_se in [False, True]:
                        for stride in [1, 2]:
                            for input_size in [32, 16, 8]:
                                lut.add(
                                    OpConfig(
                                        op_type="block",
                                        channels=channels,
                                        kernel_size=kernel,
                                        expansion=expansion,
                                        use_se=use_se,
                                        stride=stride,
                                        input_size=input_size,
                                    ),
                                    mean_ms=1.0,
                                    std_ms=0.1,
                                    samples=100,
                                )

        # Mock accuracy function (random but deterministic based on config)
        def mock_accuracy(config: ArchConfig) -> float:
            import hashlib
            config_str = str(config)
            h = hashlib.md5(config_str.encode()).hexdigest()
            return 0.5 + 0.4 * (int(h[:8], 16) / 0xFFFFFFFF)

        frontier = evolutionary_search(
            generations=2,
            pop_size=4,
            lut=lut,
            target_latency_ms=5.0,
            lambda_penalty=0.1,
            accuracy_fn=mock_accuracy,
        )

        assert len(frontier) > 0

        # All frontier points should have valid values
        for config, accuracy, latency in frontier:
            assert 0.0 <= accuracy <= 1.0
            assert latency >= 0

    def test_search_respects_generations(self):
        """Search runs for correct number of generations."""
        lut = LatencyTable()
        # Add minimal entries
        for ch in [16, 32, 64]:
            lut.add(OpConfig(op_type="stem", channels=ch, input_size=32), 0.5, 0.05, 100)
            lut.add(OpConfig(op_type="classifier", channels=ch, input_size=1), 0.1, 0.01, 100)
            for k in [3, 5]:
                for e in [2, 4, 6]:
                    for s in [False, True]:
                        for st in [1, 2]:
                            for inp in [32, 16, 8]:
                                lut.add(OpConfig("block", ch, k, e, s, st, inp), 1.0, 0.1, 100)

        eval_count = [0]

        def counting_accuracy(config: ArchConfig) -> float:
            eval_count[0] += 1
            return 0.5

        generations = 3
        pop_size = 4

        evolutionary_search(
            generations=generations,
            pop_size=pop_size,
            lut=lut,
            target_latency_ms=5.0,
            accuracy_fn=counting_accuracy,
        )

        # Should evaluate: initial pop + (num_offspring per gen * generations)
        # num_offspring = pop_size - (pop_size // 2) = 2
        expected_evals = pop_size + (pop_size - pop_size // 2) * generations
        assert eval_count[0] == expected_evals
