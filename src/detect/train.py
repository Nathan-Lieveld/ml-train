"""Detection training loop: ModelEMA, SGD + cosine LR + warmup, AMP, checkpointing."""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.detect.eval import evaluate
from src.detect.loss import DetectionLoss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LR = 0.01
DEFAULT_MOMENTUM = 0.937
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_WARMUP_EPOCHS = 3
DEFAULT_EMA_DECAY = 0.9999
DEFAULT_EMA_TAU = 2000
DEFAULT_ACCUMULATE = 1


# ---------------------------------------------------------------------------
# Model EMA
# ---------------------------------------------------------------------------


class ModelEMA:
    """Exponential Moving Average of model weights.

    Uses an exponential ramp from a lower initial decay up to *decay* over *tau* updates.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = DEFAULT_EMA_DECAY,
        tau: int = DEFAULT_EMA_TAU,
    ) -> None:
        self.ema = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    def update(self, model: nn.Module) -> None:
        """Update EMA weights from *model*."""
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        with torch.no_grad():
            for p_ema, p_model in zip(self.ema.parameters(), model.parameters()):
                p_ema.lerp_(p_model.to(p_ema.device), 1 - d)
            for b_ema, b_model in zip(self.ema.buffers(), model.buffers()):
                if b_ema.is_floating_point():
                    b_ema.lerp_(b_model.to(b_ema.device), 1 - d)
                else:
                    b_ema.copy_(b_model)

    @property
    def module(self) -> nn.Module:
        return self.ema


# ---------------------------------------------------------------------------
# Cosine LR with linear warmup
# ---------------------------------------------------------------------------


def _build_optimizer(
    model: nn.Module,
    lr: float = DEFAULT_LR,
    momentum: float = DEFAULT_MOMENTUM,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> torch.optim.SGD:
    """Build SGD optimizer with per-parameter-group weight decay handling."""
    # Separate BN/bias (no decay) from conv weights (decay)
    decay_params: list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.SGD(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        momentum=momentum,
        nesterov=True,
    )


def _cosine_lr(
    epoch: int, total_epochs: int, warmup_epochs: int, lr0: float, lrf: float = 0.01
) -> float:
    """Compute learning rate with linear warmup + cosine decay."""
    if epoch < warmup_epochs:
        return lr0 * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return lrf + (lr0 - lrf) * (1 + math.cos(math.pi * progress)) / 2


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    train_loader: DataLoader[dict[str, Any]],
    val_loader: DataLoader[dict[str, Any]] | None = None,
    epochs: int = 100,
    lr: float = DEFAULT_LR,
    device: torch.device | None = None,
    save_dir: str | Path = "runs/detect",
    batch_size: int = 16,
    accumulate: int = DEFAULT_ACCUMULATE,
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
    resume: str | Path | None = None,
    box_gain: float = 7.5,
    cls_gain: float = 0.5,
    dfl_gain: float = 1.5,
) -> dict[str, Any]:
    """Train a detection model.

    Args:
        model: DetectionModel instance.
        train_loader: Training DataLoader.
        val_loader: Optional validation DataLoader (skip eval if None).
        epochs: Total training epochs.
        lr: Initial learning rate.
        device: Compute device.
        save_dir: Directory for checkpoints.
        batch_size: Batch size (for logging only, loader handles actual batching).
        accumulate: Gradient accumulation steps.
        warmup_epochs: Linear warmup epochs.
        resume: Checkpoint path to resume from.
        box_gain: Box loss gain.
        cls_gain: Classification loss gain.
        dfl_gain: DFL loss gain.

    Returns:
        Dict with 'best_map', 'last_epoch', 'save_dir' keys.
    """
    if epochs < 0:
        raise ValueError(f"epochs must be >= 0, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if accumulate < 1:
        raise ValueError(f"accumulate must be >= 1, got {accumulate}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = _build_optimizer(model, lr=lr)
    loss_fn = DetectionLoss(model, box=box_gain, cls=cls_gain, dfl=dfl_gain)

    ema = ModelEMA(model)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    start_epoch = 0
    best_map = 0.0

    # Resume from checkpoint
    if resume is not None:
        ckpt_path = Path(resume)
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            ema.ema.load_state_dict(ckpt["ema"])
            start_epoch = ckpt["epoch"] + 1
            best_map = ckpt.get("best_map", 0.0)
            logger.info("Resumed from epoch %d (best mAP=%.4f)", start_epoch, best_map)

    if epochs == 0:
        logger.info("epochs=0, skipping training")
        return {"best_map": best_map, "last_epoch": start_epoch, "save_dir": str(save_dir)}

    logger.info(
        "Training %d epochs on %s, accumulate=%d, warmup=%d",
        epochs, device, accumulate, warmup_epochs,
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        epoch_box_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_dfl_loss = 0.0
        num_batches = 0
        num_images = 0
        optimizer_steps = 0
        data_time = 0.0
        compute_time = 0.0

        # Set LR
        current_lr = _cosine_lr(epoch, epochs, warmup_epochs, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        epoch_t0 = time.perf_counter()
        t_iter_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            t_data_end = time.perf_counter()
            data_time += t_data_end - t_iter_start

            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            num_images += images.shape[0]

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                preds = model(images)
                loss, loss_items = loss_fn(preds, targets)
                loss = loss / accumulate

            # NaN/Inf guard
            if not torch.isfinite(loss):
                logger.warning("Non-finite loss at epoch %d batch %d, skipping", epoch, batch_idx)
                optimizer.zero_grad()
                compute_time += time.perf_counter() - t_data_end
                t_iter_start = time.perf_counter()
                continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulate == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                optimizer_steps += 1

            epoch_loss += loss.item() * accumulate
            epoch_box_loss += loss_items["box"]
            epoch_cls_loss += loss_items["cls"]
            epoch_dfl_loss += loss_items["dfl"]
            num_batches += 1

            compute_time += time.perf_counter() - t_data_end
            t_iter_start = time.perf_counter()

        epoch_time = time.perf_counter() - epoch_t0
        n = max(num_batches, 1)
        avg_loss = epoch_loss / n
        throughput = num_images / epoch_time if epoch_time > 0 else 0.0

        logger.info(
            "Epoch %d/%d  lr=%.6f  loss=%.4f (box=%.4f cls=%.4f dfl=%.4f)  steps=%d",
            epoch + 1, epochs, current_lr, avg_loss,
            epoch_box_loss / n, epoch_cls_loss / n, epoch_dfl_loss / n,
            optimizer_steps,
        )
        mem_str = ""
        if device.type == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            mem_str = f"  mem={peak_mem_gb:.2f}GB"
        logger.info(
            "  %.1fs (data=%.1fs compute=%.1fs)  %.1f img/s%s",
            epoch_time, data_time, compute_time, throughput, mem_str,
        )

        # Evaluate
        metrics: dict[str, float] = {}
        if val_loader is not None:
            eval_t0 = time.perf_counter()
            metrics = evaluate(ema.module, val_loader, device)
            eval_time = time.perf_counter() - eval_t0
            current_map = metrics.get("mAP50_95", 0.0)
            logger.info(
                "  mAP50=%.4f  mAP50_95=%.4f  eval=%.1fs",
                metrics.get("mAP50", 0.0), current_map, eval_time,
            )
        else:
            current_map = 0.0

        # Save checkpoint
        ckpt_data = {
            "model": model.state_dict(),
            "ema": ema.ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_map": best_map,
            "metrics": metrics,
        }

        torch.save(ckpt_data, save_dir / "last.pt")

        if current_map > best_map:
            best_map = current_map
            ckpt_data["best_map"] = best_map
            torch.save(ckpt_data, save_dir / "best.pt")
            logger.info("  New best mAP: %.4f", best_map)

    return {"best_map": best_map, "last_epoch": epochs, "save_dir": str(save_dir)}
