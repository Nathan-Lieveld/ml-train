"""Detection loss components: CIoU, DFL, BboxLoss, TaskAlignedAssigner, DetectionLoss."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.detect.models import bbox2dist, dist2bbox, make_anchors

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

EPS = 1e-7


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    CIoU: bool = False,
    eps: float = EPS,
) -> torch.Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        box1: (N, 4) bounding boxes.
        box2: (M, 4) bounding boxes.
        xywh: If True, boxes are (cx, cy, w, h); else (x1, y1, x2, y2).
        CIoU: If True, compute Complete IoU with penalty term.
        eps: Small value to avoid division by zero.

    Returns:
        (N, M) pairwise IoU matrix.
    """
    if xywh:
        x1_1 = box1[:, 0:1] - box1[:, 2:3] / 2
        y1_1 = box1[:, 1:2] - box1[:, 3:4] / 2
        x2_1 = box1[:, 0:1] + box1[:, 2:3] / 2
        y2_1 = box1[:, 1:2] + box1[:, 3:4] / 2
        x1_2 = box2[:, 0:1] - box2[:, 2:3] / 2
        y1_2 = box2[:, 1:2] - box2[:, 3:4] / 2
        x2_2 = box2[:, 0:1] + box2[:, 2:3] / 2
        y2_2 = box2[:, 1:2] + box2[:, 3:4] / 2
    else:
        x1_1, y1_1, x2_1, y2_1 = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:, 0:1], box2[:, 1:2], box2[:, 2:3], box2[:, 3:4]

    # Pairwise intersection: (N, 1) vs (1, M) → (N, M)
    inter_x1 = torch.max(x1_1, x1_2.T)
    inter_y1 = torch.max(y1_1, y1_2.T)
    inter_x2 = torch.min(x2_1, x2_2.T)
    inter_y2 = torch.min(y2_1, y2_2.T)

    inter = (inter_x2 - inter_x1).clamp_(0) * (inter_y2 - inter_y1).clamp_(0)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)  # (N, 1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)  # (M, 1)
    union = area1 + area2.T - inter + eps

    iou = inter / union

    if CIoU:
        # Enclosing box
        cw = torch.max(x2_1, x2_2.T) - torch.min(x1_1, x1_2.T)
        ch = torch.max(y2_1, y2_2.T) - torch.min(y1_1, y1_2.T)
        c2 = cw**2 + ch**2 + eps  # enclosing diagonal squared

        # Center distance
        cx1 = (x1_1 + x2_1) / 2
        cy1 = (y1_1 + y2_1) / 2
        cx2 = (x1_2 + x2_2) / 2
        cy2 = (y1_2 + y2_2) / 2
        rho2 = (cx1 - cx2.T) ** 2 + (cy1 - cy2.T) ** 2

        # Aspect ratio penalty
        w1 = x2_1 - x1_1
        h1 = y2_1 - y1_1
        w2 = x2_2 - x1_2
        h2 = y2_2 - y1_2
        v = (4 / (torch.pi**2)) * (
            torch.atan(w1 / (h1 + eps)) - torch.atan(w2.T / (h2.T + eps))
        ) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        iou = iou - (rho2 / c2 + v * alpha)

    return iou


def _aligned_bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = False,
    CIoU: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Element-wise IoU for aligned box pairs. Both inputs must have the same shape.

    Args:
        box1: (N, 4) boxes.
        box2: (N, 4) boxes (same N).
        xywh: Box format flag.
        CIoU: Compute CIoU.

    Returns:
        (N,) IoU values.
    """
    if xywh:
        x1_1 = box1[:, 0] - box1[:, 2] / 2
        y1_1 = box1[:, 1] - box1[:, 3] / 2
        x2_1 = box1[:, 0] + box1[:, 2] / 2
        y2_1 = box1[:, 1] + box1[:, 3] / 2
        x1_2 = box2[:, 0] - box2[:, 2] / 2
        y1_2 = box2[:, 1] - box2[:, 3] / 2
        x2_2 = box2[:, 0] + box2[:, 2] / 2
        y2_2 = box2[:, 1] + box2[:, 3] / 2
    else:
        x1_1, y1_1, x2_1, y2_1 = box1.unbind(-1)
        x1_2, y1_2, x2_2, y2_2 = box2.unbind(-1)

    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)
    inter = (inter_x2 - inter_x1).clamp_(0) * (inter_y2 - inter_y1).clamp_(0)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter + eps
    iou = inter / union

    if CIoU:
        cw = torch.max(x2_1, x2_2) - torch.min(x1_1, x1_2)
        ch = torch.max(y2_1, y2_2) - torch.min(y1_1, y1_2)
        c2 = cw**2 + ch**2 + eps
        rho2 = ((x1_1 + x2_1 - x1_2 - x2_2) ** 2 + (y1_1 + y2_1 - y1_2 - y2_2) ** 2) / 4
        w1, h1 = x2_1 - x1_1, y2_1 - y1_1
        w2, h2 = x2_2 - x1_2, y2_2 - y1_2
        v = (4 / (torch.pi**2)) * (torch.atan(w1 / (h1 + eps)) - torch.atan(w2 / (h2 + eps))) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        iou = iou - (rho2 / c2 + v * alpha)

    return iou


# ---------------------------------------------------------------------------
# DFL Loss
# ---------------------------------------------------------------------------


class DFLoss(nn.Module):
    """Distribution Focal Loss on integer regression targets."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        if reg_max < 1:
            raise ValueError(f"reg_max must be >= 1, got {reg_max}")
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute DFL.

        Args:
            pred_dist: (N, reg_max) raw logits.
            target: (N,) float targets in [0, reg_max-1].

        Returns:
            Scalar loss.
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr.float() - target
        wr = 1.0 - wl
        loss = (
            F.cross_entropy(pred_dist, tl, reduction="none") * wl
            + F.cross_entropy(pred_dist, tr.clamp_(max=self.reg_max - 1), reduction="none") * wr
        )
        return loss.mean()


# ---------------------------------------------------------------------------
# Bbox Loss
# ---------------------------------------------------------------------------


class BboxLoss(nn.Module):
    """Bounding box loss: CIoU + DFL (or L1 when reg_max=1)."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bbox loss on foreground anchors.

        Args:
            pred_dist: (B, A, 4*reg_max) raw distributions.
            pred_bboxes: (B, A, 4) decoded xyxy boxes.
            anchor_points: (A, 2) anchor centers.
            target_bboxes: (B, A, 4) assigned target xyxy boxes.
            target_scores: (B, A, nc) soft assignment scores.
            target_scores_sum: Scalar normalization factor.
            fg_mask: (B, A) bool mask for foreground anchors.

        Returns:
            (iou_loss, dfl_loss) tuple.
        """
        num_fg = fg_mask.sum()
        if num_fg == 0:
            return torch.tensor(0.0, device=pred_dist.device), torch.tensor(
                0.0, device=pred_dist.device
            )

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # CIoU loss
        iou = _aligned_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL or L1
        if self.dfl_loss is not None:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            target_ltrb_fg = target_ltrb[fg_mask]
            pred_dist_fg = pred_dist[fg_mask].view(-1, self.reg_max)
            target_flat = target_ltrb_fg.view(-1)
            loss_dfl = self.dfl_loss(pred_dist_fg, target_flat)
        else:
            # L1 for reg_max=1
            target_ltrb = bbox2dist(anchor_points, target_bboxes, 0.99)
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none") * weight
            ).sum() / target_scores_sum

        return loss_iou, loss_dfl


# ---------------------------------------------------------------------------
# Task-Aligned Assigner
# ---------------------------------------------------------------------------


class TaskAlignedAssigner:
    """Assign ground-truth boxes to anchors using task-aligned metric."""

    def __init__(
        self,
        topk: int = 10,
        nc: int = 80,
        alpha: float = 0.5,
        beta: float = 6.0,
    ) -> None:
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        if nc < 1:
            raise ValueError(f"nc must be >= 1, got {nc}")
        self.topk = topk
        self.nc = nc
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def __call__(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign targets to anchors.

        Args:
            pred_scores: (B, A, nc) predicted class scores.
            pred_bboxes: (B, A, 4) predicted xyxy boxes.
            anchor_points: (A, 2) anchor center points.
            gt_labels: (B, M, 1) ground-truth class indices.
            gt_bboxes: (B, M, 4) ground-truth xyxy boxes.
            mask_gt: (B, M, 1) valid GT mask.

        Returns:
            target_labels: (B, A) assigned class labels.
            target_bboxes: (B, A, 4) assigned target boxes.
            target_scores: (B, A, nc) soft target scores.
            fg_mask: (B, A) foreground mask.
            target_gt_idx: (B, A) assigned GT index.
        """
        B, A, _ = pred_scores.shape
        _, M, _ = gt_labels.shape
        device = pred_scores.device

        if M == 0:
            return (
                torch.zeros(B, A, dtype=torch.long, device=device),
                torch.zeros(B, A, 4, device=device),
                torch.zeros(B, A, self.nc, device=device),
                torch.zeros(B, A, dtype=torch.bool, device=device),
                torch.zeros(B, A, dtype=torch.long, device=device),
            )

        # Compute pairwise IoU: (B, M, A)
        overlaps = self._compute_iou(gt_bboxes, pred_bboxes)

        # Gather predicted scores for GT classes: (B, M, A)
        ind = gt_labels.squeeze(-1).long().clamp(0, self.nc - 1)
        ind = ind.unsqueeze(2).expand(-1, -1, A)
        pred_scores_t = pred_scores.permute(0, 2, 1)  # (B, nc, A)
        align_scores = pred_scores_t.gather(1, ind)  # (B, M, A)

        # Task-aligned metric
        align_metric = align_scores.pow(self.alpha) * overlaps.pow(self.beta)
        align_metric *= mask_gt.expand(-1, -1, A)

        # Select top-k anchors per GT
        topk = min(self.topk, A)
        topk_metrics, topk_idxs = align_metric.topk(topk, dim=-1)  # (B, M, topk)

        # Build candidate mask
        candidate_mask = torch.zeros(B, M, A, dtype=torch.bool, device=device)
        candidate_mask.scatter_(2, topk_idxs, True)
        candidate_mask *= mask_gt.bool().expand(-1, -1, A)

        # Filter by center prior: anchor must be within GT box
        anchor_in_gt = self._anchor_in_gt(anchor_points, gt_bboxes)  # (B, M, A)
        candidate_mask *= anchor_in_gt

        # For each anchor, keep only the GT with highest metric (resolve conflicts)
        fg_mask = candidate_mask.any(dim=1)  # (B, A)
        metric_masked = align_metric * candidate_mask.float()
        target_gt_idx = metric_masked.argmax(dim=1)  # (B, A) — best GT per anchor

        # Gather assigned targets
        target_labels = gt_labels.squeeze(-1).long()
        target_labels = target_labels.gather(1, target_gt_idx)  # (B, A)
        target_labels *= fg_mask.long()

        target_bboxes = gt_bboxes.gather(
            1, target_gt_idx.unsqueeze(-1).expand(-1, -1, 4)
        )  # (B, A, 4)

        # Soft scores: one-hot × alignment metric
        target_scores = F.one_hot(target_labels.clamp(0), self.nc).float()  # (B, A, nc)
        metric_masked.amax(dim=1, keepdim=False)  # (B, A)
        max_metric_per_gt = metric_masked.amax(dim=2, keepdim=True)  # (B, M, 1)
        norm_metric = metric_masked / (max_metric_per_gt + EPS)
        norm_score = norm_metric.gather(1, target_gt_idx.unsqueeze(1).expand(-1, M, -1))
        norm_score = norm_score[:, 0, :]  # (B, A) — take first GT dim, others resolved
        # Re-index properly
        norm_score = torch.zeros(B, A, device=device)
        for b in range(B):
            for a in range(A):
                if fg_mask[b, a]:
                    gi = target_gt_idx[b, a]
                    denom = max_metric_per_gt[b, gi, 0] + EPS
                    norm_score[b, a] = align_metric[b, gi, a] / denom
        target_scores *= norm_score.unsqueeze(-1)
        target_scores *= fg_mask.unsqueeze(-1).float()

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

    @staticmethod
    def _compute_iou(
        gt_bboxes: torch.Tensor, pred_bboxes: torch.Tensor
    ) -> torch.Tensor:
        """Pairwise IoU: (B, M, 4) × (B, A, 4) → (B, M, A)."""
        B, M, _ = gt_bboxes.shape
        A = pred_bboxes.shape[1]

        gt = gt_bboxes.unsqueeze(2).expand(-1, -1, A, -1)  # (B, M, A, 4)
        pd = pred_bboxes.unsqueeze(1).expand(-1, M, -1, -1)  # (B, M, A, 4)

        inter_x1 = torch.max(gt[..., 0], pd[..., 0])
        inter_y1 = torch.max(gt[..., 1], pd[..., 1])
        inter_x2 = torch.min(gt[..., 2], pd[..., 2])
        inter_y2 = torch.min(gt[..., 3], pd[..., 3])
        inter = (inter_x2 - inter_x1).clamp_(0) * (inter_y2 - inter_y1).clamp_(0)

        area_gt = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
        area_pd = (pd[..., 2] - pd[..., 0]) * (pd[..., 3] - pd[..., 1])
        return inter / (area_gt + area_pd - inter + EPS)

    @staticmethod
    def _anchor_in_gt(
        anchor_points: torch.Tensor, gt_bboxes: torch.Tensor
    ) -> torch.Tensor:
        """Check if anchor centers fall inside GT boxes. Returns (B, M, A) bool."""
        B, M, _ = gt_bboxes.shape
        anchor_points.shape[0]

        ap = anchor_points.unsqueeze(0).unsqueeze(0)  # (1, 1, A, 2)
        gt = gt_bboxes.unsqueeze(2)  # (B, M, 1, 4)

        in_x = (ap[..., 0] >= gt[..., 0]) & (ap[..., 0] <= gt[..., 2])
        in_y = (ap[..., 1] >= gt[..., 1]) & (ap[..., 1] <= gt[..., 3])
        return in_x & in_y  # (B, M, A)


# ---------------------------------------------------------------------------
# Full detection loss
# ---------------------------------------------------------------------------

DEFAULT_BOX_GAIN = 7.5
DEFAULT_CLS_GAIN = 0.5
DEFAULT_DFL_GAIN = 1.5


class DetectionLoss(nn.Module):
    """Full YOLO detection loss: BCE cls + CIoU bbox + DFL."""

    def __init__(
        self,
        model: nn.Module,
        box: float = DEFAULT_BOX_GAIN,
        cls: float = DEFAULT_CLS_GAIN,
        dfl: float = DEFAULT_DFL_GAIN,
    ) -> None:
        super().__init__()
        if not hasattr(model, "detect"):
            raise ValueError("Model must have a .detect attribute (Detect head)")

        detect = model.detect
        self.nc = detect.nc
        self.reg_max = detect.reg_max
        self.no = detect.no
        self.stride = detect.stride
        self.use_dfl = self.reg_max > 1

        self.box_gain = box
        self.cls_gain = cls
        self.dfl_gain = dfl

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(self.reg_max)
        self.assigner = TaskAlignedAssigner(topk=10, nc=self.nc, alpha=0.5, beta=6.0)

        # DFL projection vector
        if self.use_dfl:
            self.proj = torch.arange(self.reg_max, dtype=torch.float)
        else:
            self.proj = torch.arange(1, dtype=torch.float)

    def forward(
        self,
        preds: list[torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute detection loss.

        Args:
            preds: List of (B, no, H, W) raw predictions from Detect head (training mode).
            targets: (N_total, 6) target tensor where each row is [batch_idx, cls, x1, y1, x2, y2].

        Returns:
            (total_loss, loss_dict) where loss_dict has 'box', 'cls', 'dfl' keys.
        """
        device = preds[0].device
        batch_size = preds[0].shape[0]

        # Flatten and split predictions
        pred_cat = torch.cat(
            [xi.view(batch_size, self.no, -1) for xi in preds], 2
        )  # (B, no, A)
        pred_distri, pred_scores = pred_cat.split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (B, A, nc)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (B, A, 4*reg_max)

        pred_scores.shape[1]

        # Generate anchors
        anchor_points, stride_tensor = make_anchors(preds, self.stride, 0.5)
        # anchor_points: (A, 2), stride_tensor: (A, 1)

        # Decode predicted bboxes
        pred_bboxes = self._bbox_decode(anchor_points, pred_distri)  # (B, A, 4) xyxy
        pred_bboxes_scaled = pred_bboxes * stride_tensor.unsqueeze(0)

        # Preprocess targets into (B, M, 5) format [cls, x1, y1, x2, y2]
        gt_labels, gt_bboxes, mask_gt = self._preprocess_targets(
            targets, batch_size, device
        )
        # gt_bboxes are already in pixel space; pass pixel-space to assigner
        # alongside pred_bboxes_scaled (also pixel space).

        # Assigner (all boxes in pixel space)
        assigned = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes_scaled.detach(),
            anchor_points * stride_tensor,  # pixel-space anchor centers
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_labels, target_bboxes, target_scores, fg_mask, _ = assigned
        # target_bboxes is (B, A, 4) in pixel space; convert to feature space
        target_bboxes = target_bboxes / stride_tensor.unsqueeze(0)

        target_scores_sum = max(target_scores.sum(), 1.0)

        # Classification loss (BCE)
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum

        # Bbox + DFL loss
        loss_box, loss_dfl = self.bbox_loss(
            pred_distri,
            pred_bboxes,  # feature-space boxes
            anchor_points,
            target_bboxes,  # feature-space targets
            target_scores,
            target_scores_sum,
            fg_mask,
        )

        # Scale by gains
        loss_box *= self.box_gain
        loss_cls *= self.cls_gain
        loss_dfl *= self.dfl_gain

        total_loss = (loss_box + loss_cls + loss_dfl) * batch_size

        return total_loss, {
            "box": loss_box.item(),
            "cls": loss_cls.item(),
            "dfl": loss_dfl.item(),
        }

    def _bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted distributions to xyxy bboxes in feature space."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            proj = self.proj.to(pred_dist.device).to(pred_dist.dtype)
            pred_dist = pred_dist.view(b, a, 4, self.reg_max).softmax(3).matmul(proj)
        return dist2bbox(pred_dist, anchor_points, xywh=False, dim=-1)

    @staticmethod
    def _preprocess_targets(
        targets: torch.Tensor, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert flat targets to batched format.

        Args:
            targets: (N, 6) [batch_idx, cls, x1, y1, x2, y2].
            batch_size: Number of images in batch.
            device: Target device.

        Returns:
            gt_labels: (B, M, 1) class indices.
            gt_bboxes: (B, M, 4) xyxy boxes.
            mask_gt: (B, M, 1) valid mask.
        """
        if targets.numel() == 0:
            return (
                torch.zeros(batch_size, 0, 1, device=device),
                torch.zeros(batch_size, 0, 4, device=device),
                torch.zeros(batch_size, 0, 1, device=device),
            )

        counts = []
        for b in range(batch_size):
            counts.append((targets[:, 0] == b).sum().item())
        max_gt = max(counts) if counts else 0

        if max_gt == 0:
            return (
                torch.zeros(batch_size, 0, 1, device=device),
                torch.zeros(batch_size, 0, 4, device=device),
                torch.zeros(batch_size, 0, 1, device=device),
            )

        gt_labels = torch.zeros(batch_size, max_gt, 1, device=device)
        gt_bboxes = torch.zeros(batch_size, max_gt, 4, device=device)
        mask_gt = torch.zeros(batch_size, max_gt, 1, device=device)

        for b in range(batch_size):
            b_mask = targets[:, 0] == b
            b_targets = targets[b_mask]
            n = b_targets.shape[0]
            if n > 0:
                gt_labels[b, :n, 0] = b_targets[:, 1]
                gt_bboxes[b, :n] = b_targets[:, 2:6]
                mask_gt[b, :n, 0] = 1.0

        return gt_labels, gt_bboxes, mask_gt
