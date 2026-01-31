"""Detection evaluation: NMS, AP computation, mAP evaluation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torchvision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------

DEFAULT_CONF_THRESHOLD = 0.001
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_MAX_DET = 300
AP_INTERP_POINTS = 101
IOU_THRESHOLDS = torch.arange(0.5, 1.0, 0.05)


def non_max_suppression(
    preds: torch.Tensor,
    conf: float = DEFAULT_CONF_THRESHOLD,
    iou: float = DEFAULT_IOU_THRESHOLD,
    max_det: int = DEFAULT_MAX_DET,
) -> list[torch.Tensor]:
    """Apply non-maximum suppression to detection predictions.

    Args:
        preds: (B, N, 4+nc) tensor — boxes (xyxy) + class scores.
        conf: Confidence threshold for filtering.
        iou: IoU threshold for NMS.
        max_det: Maximum detections per image.

    Returns:
        List of (D, 6) tensors per image: [x1, y1, x2, y2, conf, cls].
    """
    if preds.dim() != 3:
        raise ValueError(f"preds must be 3D (B, N, 4+nc), got shape {preds.shape}")

    batch_size = preds.shape[0]
    preds.shape[2] - 4
    results: list[torch.Tensor] = []

    for i in range(batch_size):
        pred = preds[i]  # (N, 4+nc)
        boxes = pred[:, :4]  # (N, 4) xyxy
        scores = pred[:, 4:]  # (N, nc) class scores

        # Max confidence across classes
        class_conf, class_idx = scores.max(dim=1)  # (N,), (N,)

        # Filter by confidence
        keep = class_conf > conf
        if not keep.any():
            results.append(torch.zeros(0, 6, device=preds.device))
            continue

        boxes = boxes[keep]
        class_conf = class_conf[keep]
        class_idx = class_idx[keep]

        # Apply batched NMS (offset boxes by class for per-class NMS)
        nms_idx = torchvision.ops.batched_nms(
            boxes, class_conf, class_idx, iou
        )

        # Limit detections
        if nms_idx.shape[0] > max_det:
            nms_idx = nms_idx[:max_det]

        det = torch.cat(
            [boxes[nms_idx], class_conf[nms_idx].unsqueeze(1), class_idx[nms_idx].float().unsqueeze(1)],
            dim=1,
        )  # (D, 6)
        results.append(det)

    return results


# ---------------------------------------------------------------------------
# Average precision
# ---------------------------------------------------------------------------


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP using 101-point interpolation.

    Args:
        recall: (N,) sorted recall values.
        precision: (N,) corresponding precision values.

    Returns:
        Average precision value.
    """
    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # 101-point interpolation
    x = np.linspace(0, 1, AP_INTERP_POINTS)
    ap = np.mean(np.interp(x, mrec, mpre))
    return float(ap)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def _match_predictions(
    pred_boxes: torch.Tensor,
    pred_cls: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_cls: torch.Tensor,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match predictions to ground truths at a given IoU threshold.

    Returns:
        tp: (N_pred,) bool array — True for true positives.
        conf: (N_pred,) confidence array.
    """
    n_pred = pred_boxes.shape[0]
    n_gt = gt_boxes.shape[0]

    if n_pred == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)

    tp = np.zeros(n_pred, dtype=bool)

    if n_gt == 0:
        return tp, np.ones(n_pred)

    # Compute IoU matrix (N_pred, N_gt)
    iou_matrix = torchvision.ops.box_iou(pred_boxes, gt_boxes).cpu().numpy()

    matched_gt: set[int] = set()
    for pi in range(n_pred):
        if pred_cls[pi].item() not in gt_cls.tolist():
            continue

        # Find best matching GT of same class
        best_iou = 0.0
        best_gi = -1
        for gi in range(n_gt):
            if gi in matched_gt:
                continue
            if gt_cls[gi].item() != pred_cls[pi].item():
                continue
            if iou_matrix[pi, gi] > best_iou:
                best_iou = iou_matrix[pi, gi]
                best_gi = gi

        if best_iou >= iou_threshold and best_gi >= 0:
            tp[pi] = True
            matched_gt.add(best_gi)

    return tp, np.ones(n_pred)


def evaluate(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    conf: float = DEFAULT_CONF_THRESHOLD,
    iou_nms: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, float]:
    """Evaluate detection model: compute mAP@0.5 and mAP@0.5:0.95.

    Args:
        model: Detection model in eval mode.
        dataloader: DataLoader yielding dicts with 'images' and 'targets'.
        device: Compute device.
        conf: NMS confidence threshold.
        iou_nms: NMS IoU threshold.

    Returns:
        Dict with 'mAP50' and 'mAP50_95' keys.
    """
    model.eval()
    all_tp: dict[float, list[np.ndarray]] = {t.item(): [] for t in IOU_THRESHOLDS}
    all_conf: list[np.ndarray] = []
    total_gt = 0

    thresholds = [t.item() for t in IOU_THRESHOLDS]

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            targets = batch["targets"]

            preds = model(images)  # (B, A, 4+nc)
            detections = non_max_suppression(preds, conf=conf, iou=iou_nms)

            batch_size = images.shape[0]
            for i in range(batch_size):
                det = detections[i]
                # Get GT for this image
                gt_mask = targets[:, 0] == i
                gt = targets[gt_mask]

                if gt.shape[0] > 0:
                    gt_boxes = gt[:, 2:6].to(device)
                    gt_cls = gt[:, 1].to(device)
                    total_gt += gt.shape[0]
                else:
                    gt_boxes = torch.zeros(0, 4, device=device)
                    gt_cls = torch.zeros(0, device=device)

                if det.shape[0] == 0:
                    for t in thresholds:
                        all_tp[t].append(np.zeros(0, dtype=bool))
                    all_conf.append(np.zeros(0))
                    continue

                pred_boxes = det[:, :4]
                pred_conf = det[:, 4]
                pred_cls = det[:, 5]

                all_conf.append(pred_conf.cpu().numpy())

                for t in thresholds:
                    tp, _ = _match_predictions(
                        pred_boxes, pred_cls, gt_boxes, gt_cls, t
                    )
                    all_tp[t].append(tp)

    if total_gt == 0:
        return {"mAP50": 0.0, "mAP50_95": 0.0}

    # Concatenate all predictions
    if all_conf:
        all_conf_arr = np.concatenate(all_conf)
    else:
        return {"mAP50": 0.0, "mAP50_95": 0.0}

    # Sort by confidence (descending)
    sort_idx = np.argsort(-all_conf_arr)

    aps: dict[float, float] = {}
    for t in thresholds:
        if all_tp[t]:
            tp_arr = np.concatenate(all_tp[t])
        else:
            tp_arr = np.zeros(0, dtype=bool)

        tp_sorted = tp_arr[sort_idx] if tp_arr.size > 0 else tp_arr
        tp_cum = np.cumsum(tp_sorted)
        fp_cum = np.cumsum(~tp_sorted)

        recall = tp_cum / total_gt
        precision = tp_cum / (tp_cum + fp_cum)

        aps[t] = compute_ap(recall, precision)

    map50 = aps.get(0.5, 0.0)
    map50_95 = float(np.mean(list(aps.values())))

    return {"mAP50": map50, "mAP50_95": map50_95}
