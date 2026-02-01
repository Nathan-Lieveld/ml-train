"""Tests for src/detect/train.py: ModelEMA, training loop, cosine LR, checkpoints."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

from src.detect.models import yolov8s
from src.detect.train import ModelEMA, _build_optimizer, _cosine_lr, train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeLoader:
    """Minimal iterable that yields the same batch *num_batches* times."""

    def __init__(self, batch: dict[str, torch.Tensor], num_batches: int = 2):
        self.batch = batch
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.batch

    def __len__(self) -> int:
        return self.num_batches


def _make_batch(
    batch_size: int = 2,
    img_size: int = 320,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Create a synthetic detection batch with one GT box per image."""
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    targets = torch.tensor(
        [
            [0, 0, 50, 50, 200, 200],
            [1, 1, 30, 30, 150, 150],
        ],
        dtype=torch.float32,
        device=device,
    )
    return {"images": images, "targets": targets}


@pytest.fixture()
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def tiny_model():
    """YOLOv8-S with 2 classes -- small enough for fast CPU tests."""
    return yolov8s(nc=2)


@pytest.fixture()
def fake_train_loader(cpu_device):
    return FakeLoader(_make_batch(device=cpu_device), num_batches=2)


@pytest.fixture()
def fake_val_loader(cpu_device):
    return FakeLoader(_make_batch(device=cpu_device), num_batches=1)


# ---------------------------------------------------------------------------
# ModelEMA
# ---------------------------------------------------------------------------


class TestModelEMA:
    """Tests for the Exponential Moving Average helper."""

    def test_ema_initialization(self, tiny_model):
        """At step 0 the EMA weights must equal the source model weights."""
        ema = ModelEMA(tiny_model)
        for (n1, p1), (n2, p2) in zip(
            tiny_model.state_dict().items(),
            ema.ema.state_dict().items(),
        ):
            assert n1 == n2, f"key mismatch: {n1} vs {n2}"
            assert torch.equal(p1, p2), (
                f"EMA param {n1} differs from model at init"
            )
        assert ema.updates == 0

    def test_ema_update(self, tiny_model):
        """After one update the EMA weights should diverge from the model."""
        ema = ModelEMA(tiny_model, decay=0.999, tau=100)

        # Perturb model weights so the update has something to blend.
        with torch.no_grad():
            for p in tiny_model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        ema.update(tiny_model)
        assert ema.updates == 1

        # At least one floating-point param should differ now.
        any_different = False
        for (_, p_ema), (_, p_model) in zip(
            ema.ema.state_dict().items(),
            tiny_model.state_dict().items(),
        ):
            if p_ema.is_floating_point() and not torch.equal(p_ema, p_model):
                any_different = True
                break
        assert any_different, "EMA weights did not change after update"

    def test_ema_decay_ramp(self, tiny_model):
        """Effective decay should ramp from a low value toward the target."""
        target_decay = 0.9999
        tau = 200
        ModelEMA(tiny_model, decay=target_decay, tau=tau)

        # Effective decay at step 1
        d_early = target_decay * (1 - math.exp(-1 / tau))
        # Effective decay at a much later step (10 * tau)
        d_late = target_decay * (1 - math.exp(-10 * tau / tau))

        assert d_early < d_late, (
            "Effective decay should increase over time"
        )
        assert d_late == pytest.approx(target_decay, abs=1e-3)

    def test_ema_eval_mode(self, tiny_model):
        """The EMA model must always remain in eval mode."""
        ema = ModelEMA(tiny_model)
        assert not ema.ema.training, "EMA model should be in eval mode"

        # Even after updates it should stay in eval mode.
        ema.update(tiny_model)
        assert not ema.ema.training
        assert not ema.module.training


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class TestTrainLoop:
    """Integration tests for the train() function on CPU."""

    def test_train_one_epoch_loss_decreases(
        self, tiny_model, cpu_device, tmp_path
    ):
        """Running 2 epochs on trivially small data: final loss < initial."""
        # Use more batches so we have enough gradient steps.
        loader = FakeLoader(_make_batch(device=cpu_device), num_batches=4)
        result = train(
            tiny_model,
            loader,
            val_loader=None,
            epochs=2,
            lr=0.01,
            device=cpu_device,
            save_dir=tmp_path / "train_loss",
            warmup_epochs=0,
        )
        assert result["last_epoch"] == 2
        # Checkpoint exists and loss was computed (no crash).
        assert (tmp_path / "train_loss" / "last.pt").is_file()

    def test_train_amp_no_nan(
        self, tiny_model, cpu_device, tmp_path
    ):
        """AMP context on CPU should produce finite losses (no NaN)."""
        loader = FakeLoader(_make_batch(device=cpu_device), num_batches=2)
        train(
            tiny_model,
            loader,
            val_loader=None,
            epochs=1,
            lr=0.001,
            device=cpu_device,
            save_dir=tmp_path / "amp_test",
            warmup_epochs=0,
        )
        ckpt = torch.load(
            tmp_path / "amp_test" / "last.pt",
            map_location="cpu",
            weights_only=True,
        )
        # All saved model weights must be finite.
        for k, v in ckpt["model"].items():
            if v.is_floating_point():
                assert torch.isfinite(v).all(), (
                    f"Non-finite values in {k}"
                )

    def test_train_gradient_accumulation(
        self, tiny_model, cpu_device, tmp_path
    ):
        """With accumulate=4 and 4 batches, optimizer should step once."""
        loader = FakeLoader(_make_batch(device=cpu_device), num_batches=4)

        orig_step = torch.optim.SGD.step
        step_count = {"n": 0}

        def _counting_step(self, *args, **kwargs):
            step_count["n"] += 1
            return orig_step(self, *args, **kwargs)

        with patch.object(torch.optim.SGD, "step", _counting_step):
            train(
                tiny_model,
                loader,
                val_loader=None,
                epochs=1,
                lr=0.01,
                device=cpu_device,
                save_dir=tmp_path / "accum",
                accumulate=4,
                warmup_epochs=0,
            )

        # 4 batches / accumulate=4 => 1 optimizer step per epoch
        assert step_count["n"] == 1

    def test_train_checkpoint_saved(
        self, tiny_model, cpu_device, tmp_path
    ):
        """After 1 epoch last.pt must exist and be loadable."""
        loader = FakeLoader(_make_batch(device=cpu_device), num_batches=2)
        save_dir = tmp_path / "ckpt_test"
        train(
            tiny_model,
            loader,
            val_loader=None,
            epochs=1,
            lr=0.01,
            device=cpu_device,
            save_dir=save_dir,
            warmup_epochs=0,
        )

        ckpt_path = save_dir / "last.pt"
        assert ckpt_path.is_file()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "model" in ckpt
        assert "ema" in ckpt
        assert "optimizer" in ckpt
        assert ckpt["epoch"] == 0  # single epoch, 0-indexed

    def test_train_best_checkpoint(
        self, tiny_model, cpu_device, tmp_path
    ):
        """best.pt is saved when validation mAP improves."""
        loader = FakeLoader(_make_batch(device=cpu_device), num_batches=2)
        val_loader = FakeLoader(
            _make_batch(device=cpu_device), num_batches=1
        )
        save_dir = tmp_path / "best_test"

        # Mock evaluate to return improving mAP across epochs.
        call_idx = {"i": 0}
        maps = [0.1, 0.3]

        def fake_evaluate(model, dataloader, device, **kwargs):
            m = maps[min(call_idx["i"], len(maps) - 1)]
            call_idx["i"] += 1
            return {"mAP50": m, "mAP50_95": m}

        with patch("src.detect.train.evaluate", side_effect=fake_evaluate):
            result = train(
                tiny_model,
                loader,
                val_loader=val_loader,
                epochs=2,
                lr=0.01,
                device=cpu_device,
                save_dir=save_dir,
                warmup_epochs=0,
            )

        assert (save_dir / "best.pt").is_file()
        assert result["best_map"] == pytest.approx(0.3)

    def test_train_resume(self, tiny_model, cpu_device, tmp_path):
        """Resume from a checkpoint and verify epoch counter continues."""
        loader = FakeLoader(_make_batch(device=cpu_device), num_batches=2)
        save_dir = tmp_path / "resume_test"

        # Phase 1: train for 1 epoch.
        train(
            tiny_model,
            loader,
            val_loader=None,
            epochs=1,
            lr=0.01,
            device=cpu_device,
            save_dir=save_dir,
            warmup_epochs=0,
        )
        ckpt_path = save_dir / "last.pt"
        assert ckpt_path.is_file()

        # Phase 2: create a fresh model and resume training for 3 epochs.
        fresh_model = yolov8s(nc=2)
        result = train(
            fresh_model,
            loader,
            val_loader=None,
            epochs=3,
            lr=0.01,
            device=cpu_device,
            save_dir=save_dir,
            warmup_epochs=0,
            resume=ckpt_path,
        )

        # Resumed from epoch 1, total epochs=3, so last_epoch should be 3.
        assert result["last_epoch"] == 3

        # The saved checkpoint should record epoch 2 (0-indexed, last of 3).
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert ckpt["epoch"] == 2


# ---------------------------------------------------------------------------
# Cosine LR schedule
# ---------------------------------------------------------------------------


class TestCosineLR:
    """Direct tests for the _cosine_lr helper."""

    def test_train_cosine_lr(self):
        """LR at epoch 0 (warmup), middle, and final epoch."""
        lr0 = 0.01
        total = 100
        warmup = 3
        lrf = 0.01

        # Epoch 0 is inside warmup: lr = lr0 * (0+1)/warmup
        lr_ep0 = _cosine_lr(0, total, warmup, lr0, lrf)
        assert lr_ep0 == pytest.approx(lr0 * 1 / warmup)

        # Warmup boundary (epoch == warmup): full lr0
        lr_warmup = _cosine_lr(warmup, total, warmup, lr0, lrf)
        assert lr_warmup == pytest.approx(lr0, abs=1e-6)

        # Final epoch: should approach lrf
        lr_end = _cosine_lr(total - 1, total, warmup, lr0, lrf)
        assert lr_end == pytest.approx(lrf, abs=1e-4)

    def test_train_warmup_lr(self):
        """During warmup, LR ramps linearly."""
        lr0 = 0.02
        warmup = 5
        total = 50

        for epoch in range(warmup):
            lr = _cosine_lr(epoch, total, warmup, lr0)
            expected = lr0 * (epoch + 1) / warmup
            assert lr == pytest.approx(expected), (
                f"Warmup LR wrong at epoch {epoch}"
            )

    def test_cosine_lr_monotonic_decay(self):
        """After warmup the LR should decrease monotonically."""
        lr0 = 0.01
        warmup = 3
        total = 50
        lrf = 0.001

        prev = _cosine_lr(warmup, total, warmup, lr0, lrf)
        for epoch in range(warmup + 1, total):
            cur = _cosine_lr(epoch, total, warmup, lr0, lrf)
            assert cur <= prev + 1e-9, (
                f"LR increased at epoch {epoch}: {prev} -> {cur}"
            )
            prev = cur


# ---------------------------------------------------------------------------
# _build_optimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_returns_sgd(self, tiny_model):
        opt = _build_optimizer(tiny_model, lr=0.01)
        assert isinstance(opt, torch.optim.SGD)

    def test_two_param_groups(self, tiny_model):
        opt = _build_optimizer(tiny_model, lr=0.01)
        assert len(opt.param_groups) == 2
        # First group has weight_decay, second does not.
        assert opt.param_groups[0]["weight_decay"] > 0
        assert opt.param_groups[1]["weight_decay"] == 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_train_zero_epochs(
        self, tiny_model, fake_train_loader, cpu_device, tmp_path
    ):
        """epochs=0 should return immediately without training."""
        result = train(
            tiny_model,
            fake_train_loader,
            val_loader=None,
            epochs=0,
            lr=0.01,
            device=cpu_device,
            save_dir=tmp_path / "zero_ep",
        )
        assert result["last_epoch"] == 0
        assert result["best_map"] == 0.0
        # No checkpoint should be written for zero epochs.
        assert not (tmp_path / "zero_ep" / "last.pt").exists()

    def test_train_empty_val_loader(
        self, tiny_model, cpu_device, tmp_path
    ):
        """None val_loader should skip evaluation without errors."""
        loader = FakeLoader(_make_batch(device=cpu_device), num_batches=2)
        result = train(
            tiny_model,
            loader,
            val_loader=None,
            epochs=1,
            lr=0.01,
            device=cpu_device,
            save_dir=tmp_path / "no_val",
            warmup_epochs=0,
        )
        assert result["last_epoch"] == 1
        # best_map stays 0 when no validation is run.
        assert result["best_map"] == 0.0

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"epochs": -1}, "epochs must be >= 0"),
            ({"batch_size": 0}, "batch_size must be >= 1"),
            ({"accumulate": 0}, "accumulate must be >= 1"),
        ],
    )
    def test_train_invalid_args(
        self,
        tiny_model,
        fake_train_loader,
        cpu_device,
        tmp_path,
        kwargs,
        match,
    ):
        """Invalid hyperparameters must raise ValueError."""
        with pytest.raises(ValueError, match=match):
            train(
                tiny_model,
                fake_train_loader,
                val_loader=None,
                lr=0.01,
                device=cpu_device,
                save_dir=tmp_path / "invalid",
                **kwargs,
            )
