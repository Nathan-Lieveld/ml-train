"""Comprehensive tests for the detection loss module (src/detect/loss.py).

Covers: bbox_iou, _aligned_bbox_iou, DFLoss, BboxLoss,
        TaskAlignedAssigner, DetectionLoss.
All tests are CPU-only.
"""

import pytest
import torch

from src.detect.loss import (
    BboxLoss,
    DetectionLoss,
    DFLoss,
    TaskAlignedAssigner,
    _aligned_bbox_iou,
    bbox_iou,
)
from src.detect.models import dist2bbox, yolov8s

# ------------------------------------------------------------------ #
# IoU tests                                                          #
# ------------------------------------------------------------------ #


class TestBboxIou:
    """Tests for pairwise bbox_iou."""

    def test_bbox_iou_identical(self):
        """Identical boxes produce IoU = 1.0."""
        box = torch.tensor([[5.0, 5.0, 4.0, 4.0]])  # xywh
        iou = bbox_iou(box, box, xywh=True, CIoU=False)
        assert iou.shape == (1, 1)
        assert torch.allclose(iou, torch.ones(1, 1), atol=1e-5)

    def test_bbox_iou_identical_xyxy(self):
        """Identical xyxy boxes produce IoU = 1.0."""
        box = torch.tensor([[1.0, 2.0, 5.0, 6.0]])
        iou = bbox_iou(box, box, xywh=False, CIoU=False)
        assert torch.allclose(iou, torch.ones(1, 1), atol=1e-5)

    def test_bbox_iou_no_overlap(self):
        """Disjoint boxes produce IoU = 0.0."""
        box1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])  # xywh
        box2 = torch.tensor([[10.0, 10.0, 1.0, 1.0]])
        iou = bbox_iou(box1, box2, xywh=True, CIoU=False)
        assert torch.allclose(iou, torch.zeros(1, 1), atol=1e-5)

    def test_bbox_iou_partial(self):
        """Known partial overlap: two unit boxes offset by 0.5."""
        # xyxy format: box1 = [0,0,1,1], box2 = [0.5,0.5,1.5,1.5]
        # intersection = 0.5*0.5 = 0.25
        # union = 1 + 1 - 0.25 = 1.75
        # IoU = 0.25 / 1.75 ~ 0.142857
        box1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        box2 = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
        iou = bbox_iou(box1, box2, xywh=False, CIoU=False)
        expected = 0.25 / 1.75
        assert abs(iou.item() - expected) < 1e-4

    def test_bbox_iou_ciou_penalty(self):
        """CIoU < plain IoU when aspect ratios differ."""
        # Same center, same area, different aspect ratios
        box1 = torch.tensor([[5.0, 5.0, 4.0, 2.0]])  # wide
        box2 = torch.tensor([[5.0, 5.0, 2.0, 4.0]])  # tall
        iou_plain = bbox_iou(box1, box2, xywh=True, CIoU=False)
        iou_ciou = bbox_iou(box1, box2, xywh=True, CIoU=True)
        assert iou_ciou.item() < iou_plain.item()

    def test_bbox_iou_batch(self):
        """(N, 4) x (M, 4) -> (N, M) output shape."""
        N, M = 5, 7
        box1 = torch.rand(N, 4)
        box2 = torch.rand(M, 4)
        iou = bbox_iou(box1, box2, xywh=True, CIoU=False)
        assert iou.shape == (N, M)

    def test_bbox_iou_symmetry(self):
        """IoU(A, B) should equal IoU(B, A) transposed."""
        box1 = torch.tensor([[0.0, 0.0, 2.0, 3.0]])
        box2 = torch.tensor([[1.0, 1.0, 3.0, 4.0]])
        iou_ab = bbox_iou(box1, box2, xywh=False, CIoU=False)
        iou_ba = bbox_iou(box2, box1, xywh=False, CIoU=False)
        assert torch.allclose(iou_ab, iou_ba.T, atol=1e-5)


class TestAlignedBboxIou:
    """Tests for element-wise _aligned_bbox_iou."""

    def test_identical_xyxy(self):
        """Identical aligned boxes produce IoU = 1.0."""
        box = torch.tensor([[1.0, 2.0, 5.0, 6.0]])
        iou = _aligned_bbox_iou(box, box, xywh=False, CIoU=False)
        assert iou.shape == (1,)
        assert torch.allclose(iou, torch.ones(1), atol=1e-5)

    def test_ciou_identical(self):
        """CIoU of identical boxes is also 1.0 (no penalty)."""
        box = torch.tensor([[1.0, 2.0, 5.0, 6.0]])
        iou = _aligned_bbox_iou(box, box, xywh=False, CIoU=True)
        assert torch.allclose(iou, torch.ones(1), atol=1e-5)

    def test_batch_element_wise(self):
        """Output length equals input batch dimension."""
        N = 10
        box1 = torch.rand(N, 4)
        box2 = torch.rand(N, 4)
        # Make valid xyxy by sorting
        box1[:, 2:] = box1[:, :2] + box1[:, 2:].abs() + 0.1
        box2[:, 2:] = box2[:, :2] + box2[:, 2:].abs() + 0.1
        iou = _aligned_bbox_iou(box1, box2, xywh=False, CIoU=False)
        assert iou.shape == (N,)


# ------------------------------------------------------------------ #
# DFLoss tests                                                       #
# ------------------------------------------------------------------ #


class TestDFLoss:
    """Tests for DFLoss (Distribution Focal Loss)."""

    def test_dfloss_gradient_exists(self):
        """loss.backward() populates gradients on pred_dist."""
        reg_max = 16
        dfl = DFLoss(reg_max=reg_max)
        pred = torch.randn(8, reg_max, requires_grad=True)
        target = torch.rand(8) * (reg_max - 1)
        loss = dfl(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
        assert not torch.all(pred.grad == 0)

    def test_dfloss_zero_on_integer_target(self):
        """Target exactly on a grid integer -> loss is near zero."""
        reg_max = 16
        dfl = DFLoss(reg_max=reg_max)
        target_idx = 5
        # pred_dist with huge logit at exactly target_idx
        pred = torch.full((4, reg_max), -100.0)
        pred[:, target_idx] = 100.0
        target = torch.full((4,), float(target_idx))
        loss = dfl(pred, target)
        assert loss.item() < 1e-3

    def test_dfloss_reg_max_boundary(self):
        """Targets at 0 and reg_max-1 produce finite loss."""
        reg_max = 16
        dfl = DFLoss(reg_max=reg_max)
        pred = torch.randn(2, reg_max)
        target = torch.tensor([0.0, reg_max - 1.0])
        loss = dfl(pred, target)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_dfloss_invalid_reg_max(self):
        """reg_max < 1 raises ValueError."""
        with pytest.raises(ValueError, match="reg_max"):
            DFLoss(reg_max=0)

    def test_dfloss_scalar_output(self):
        """Output is a scalar (0-d tensor)."""
        dfl = DFLoss(reg_max=8)
        pred = torch.randn(3, 8)
        target = torch.rand(3) * 7
        loss = dfl(pred, target)
        assert loss.dim() == 0


# ------------------------------------------------------------------ #
# BboxLoss tests                                                     #
# ------------------------------------------------------------------ #


class TestBboxLoss:
    """Tests for BboxLoss (CIoU + DFL or L1)."""

    @staticmethod
    def _make_bbox_inputs(
        batch_size=1, num_anchors=16, reg_max=16, nc=2
    ):
        """Build synthetic inputs for BboxLoss.forward."""
        anchor_points = torch.rand(num_anchors, 2) * 10 + 1
        # Predicted distributions
        pred_dist = torch.randn(
            batch_size, num_anchors, 4 * reg_max
        )
        # Decode to xyxy boxes
        if reg_max > 1:
            proj = torch.arange(reg_max, dtype=torch.float)
            dist_vals = (
                pred_dist.view(batch_size, num_anchors, 4, reg_max)
                .softmax(3)
                .matmul(proj)
            )
        else:
            dist_vals = pred_dist
        pred_bboxes = dist2bbox(
            dist_vals, anchor_points, xywh=False, dim=-1
        )
        # Targets: use pred boxes + small noise
        target_bboxes = pred_bboxes.detach().clone()
        target_bboxes += torch.randn_like(target_bboxes) * 0.1
        # Fix xyxy ordering
        for b in range(batch_size):
            for a in range(num_anchors):
                for d in range(2):
                    mn = min(
                        target_bboxes[b, a, d].item(),
                        target_bboxes[b, a, d + 2].item(),
                    )
                    mx = max(
                        target_bboxes[b, a, d].item(),
                        target_bboxes[b, a, d + 2].item(),
                    )
                    target_bboxes[b, a, d] = mn
                    target_bboxes[b, a, d + 2] = mx

        # Scores and mask: all foreground
        target_scores = torch.rand(batch_size, num_anchors, nc)
        fg_mask = torch.ones(
            batch_size, num_anchors, dtype=torch.bool
        )
        target_scores_sum = target_scores.sum().clamp(min=1.0)

        return (
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
        )

    def test_bboxloss_ciou_path(self):
        """reg_max=16 uses CIoU + DFL branch."""
        reg_max = 16
        bbox_loss = BboxLoss(reg_max=reg_max)
        assert bbox_loss.dfl_loss is not None
        inputs = self._make_bbox_inputs(reg_max=reg_max)
        iou_loss, dfl_loss = bbox_loss(*inputs)
        assert torch.isfinite(iou_loss)
        assert torch.isfinite(dfl_loss)
        assert iou_loss.item() >= 0
        assert dfl_loss.item() >= 0

    def test_bboxloss_l1_path(self):
        """reg_max=1 uses L1 branch (no DFL)."""
        reg_max = 1
        bbox_loss = BboxLoss(reg_max=reg_max)
        assert bbox_loss.dfl_loss is None
        inputs = self._make_bbox_inputs(reg_max=reg_max, nc=2)
        iou_loss, dfl_loss = bbox_loss(*inputs)
        assert torch.isfinite(iou_loss)
        assert torch.isfinite(dfl_loss)

    def test_bboxloss_no_foreground(self):
        """Zero foreground anchors produce loss = 0."""
        bbox_loss = BboxLoss(reg_max=16)
        B, A, reg_max, nc = 1, 8, 16, 2
        pred_dist = torch.randn(B, A, 4 * reg_max)
        pred_bboxes = torch.randn(B, A, 4)
        anchor_points = torch.rand(A, 2)
        target_bboxes = torch.zeros(B, A, 4)
        target_scores = torch.zeros(B, A, nc)
        target_scores_sum = torch.tensor(1.0)
        fg_mask = torch.zeros(B, A, dtype=torch.bool)
        iou_loss, dfl_loss = bbox_loss(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
        )
        assert iou_loss.item() == 0.0
        assert dfl_loss.item() == 0.0


# ------------------------------------------------------------------ #
# TaskAlignedAssigner tests                                          #
# ------------------------------------------------------------------ #


class TestTaskAlignedAssigner:
    """Tests for TaskAlignedAssigner."""

    @staticmethod
    def _make_assigner_inputs(
        batch_size=1, num_anchors=50, num_gt=1, nc=2
    ):
        """Build synthetic inputs for the assigner."""
        pred_scores = torch.rand(batch_size, num_anchors, nc)
        # xyxy boxes in a reasonable range
        xy1 = torch.rand(batch_size, num_anchors, 2) * 5
        xy2 = xy1 + torch.rand(batch_size, num_anchors, 2) * 5 + 0.5
        pred_bboxes = torch.cat([xy1, xy2], dim=-1)

        anchor_points = torch.rand(num_anchors, 2) * 10

        gt_labels = torch.zeros(batch_size, num_gt, 1)
        gt_x1y1 = torch.rand(batch_size, num_gt, 2) * 3
        gt_x2y2 = gt_x1y1 + torch.rand(batch_size, num_gt, 2) * 4 + 1
        gt_bboxes = torch.cat([gt_x1y1, gt_x2y2], dim=-1)
        mask_gt = torch.ones(batch_size, num_gt, 1)

        return (
            pred_scores,
            pred_bboxes,
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

    def test_assigner_single_gt(self):
        """One GT with topk=10: some anchors get assigned."""
        assigner = TaskAlignedAssigner(topk=10, nc=2)
        # Use anchors placed inside the GT box to ensure assignment
        B, A, nc = 1, 50, 2
        pred_scores = torch.rand(B, A, nc) * 0.5 + 0.1
        gt_bboxes = torch.tensor([[[2.0, 2.0, 8.0, 8.0]]])
        # Place anchors in a grid inside the GT box
        ax = torch.linspace(2.5, 7.5, 10)
        ay = torch.linspace(2.5, 7.5, 5)
        grid_y, grid_x = torch.meshgrid(ay, ax, indexing="ij")
        anchor_points = torch.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1
        )
        # Pred bboxes around the anchor points
        pred_bboxes = torch.cat(
            [anchor_points - 1.0, anchor_points + 1.0], dim=-1
        ).unsqueeze(0)

        gt_labels = torch.zeros(B, 1, 1)
        mask_gt = torch.ones(B, 1, 1)

        out = assigner(
            pred_scores,
            pred_bboxes,
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_labels, target_bboxes, target_scores, fg_mask, _ = out
        assert fg_mask.shape == (B, A)
        assert fg_mask.sum().item() > 0
        assert target_bboxes.shape == (B, A, 4)
        assert target_scores.shape == (B, A, nc)

    def test_assigner_no_gt(self):
        """Zero GT boxes: all anchors unassigned."""
        assigner = TaskAlignedAssigner(topk=10, nc=2)
        B, A, nc = 1, 20, 2
        pred_scores = torch.rand(B, A, nc)
        pred_bboxes = torch.rand(B, A, 4)
        anchor_points = torch.rand(A, 2) * 10
        gt_labels = torch.zeros(B, 0, 1)
        gt_bboxes = torch.zeros(B, 0, 4)
        mask_gt = torch.zeros(B, 0, 1)

        out = assigner(
            pred_scores,
            pred_bboxes,
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        _, _, _, fg_mask, _ = out
        assert fg_mask.sum().item() == 0

    def test_assigner_conflict_resolution(self):
        """Two GTs competing for same anchor: higher metric wins."""
        assigner = TaskAlignedAssigner(topk=10, nc=2)
        B, A, nc = 1, 20, 2

        # Put anchor at (5, 5); both GTs cover it
        anchor_points = torch.full((A, 2), 5.0)
        pred_bboxes = torch.zeros(B, A, 4)
        pred_bboxes[:, :, 0] = 3.0
        pred_bboxes[:, :, 1] = 3.0
        pred_bboxes[:, :, 2] = 7.0
        pred_bboxes[:, :, 3] = 7.0

        # GT0: class 0, smaller overlap box
        # GT1: class 1, larger overlap box (should dominate)
        gt_labels = torch.tensor([[[0.0], [1.0]]])
        gt_bboxes = torch.tensor([
            [[4.0, 4.0, 6.0, 6.0], [2.0, 2.0, 8.0, 8.0]]
        ])
        mask_gt = torch.ones(B, 2, 1)

        # Scores: higher for class 1 to boost GT1 metric
        pred_scores = torch.zeros(B, A, nc)
        pred_scores[:, :, 1] = 0.9
        pred_scores[:, :, 0] = 0.1

        out = assigner(
            pred_scores,
            pred_bboxes,
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_labels, _, _, fg_mask, target_gt_idx = out

        # Where assigned, the GT with higher alignment should win
        assigned = fg_mask[0]
        if assigned.any():
            # GT1 (index 1) should be chosen because its metric is
            # higher (better IoU and higher predicted score for its
            # class)
            assert (target_gt_idx[0, assigned] == 1).all()

    def test_assigner_deterministic(self):
        """Same input twice produces identical output."""
        assigner = TaskAlignedAssigner(topk=5, nc=2)
        inputs = self._make_assigner_inputs(
            batch_size=1, num_anchors=30, num_gt=2
        )
        out1 = assigner(*inputs)
        out2 = assigner(*inputs)
        for t1, t2 in zip(out1, out2):
            assert torch.equal(t1, t2)

    def test_assigner_invalid_topk(self):
        """topk < 1 raises ValueError."""
        with pytest.raises(ValueError, match="topk"):
            TaskAlignedAssigner(topk=0, nc=2)

    def test_assigner_invalid_nc(self):
        """nc < 1 raises ValueError."""
        with pytest.raises(ValueError, match="nc"):
            TaskAlignedAssigner(topk=10, nc=0)

    def test_assigner_output_shapes(self):
        """Check all 5 output tensor shapes."""
        assigner = TaskAlignedAssigner(topk=10, nc=3)
        B, A, nc, M = 2, 40, 3, 3
        inputs = self._make_assigner_inputs(
            batch_size=B, num_anchors=A, num_gt=M, nc=nc
        )
        (
            target_labels,
            target_bboxes,
            target_scores,
            fg_mask,
            target_gt_idx,
        ) = assigner(*inputs)
        assert target_labels.shape == (B, A)
        assert target_bboxes.shape == (B, A, 4)
        assert target_scores.shape == (B, A, nc)
        assert fg_mask.shape == (B, A)
        assert target_gt_idx.shape == (B, A)


# ------------------------------------------------------------------ #
# DetectionLoss tests                                                #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def yolo_model_and_preds():
    """Create a small YOLOv8s model (nc=2) and cache preds.

    Uses (1, 3, 320, 320) input for a valid multi-scale output.
    """
    torch.manual_seed(42)
    model = yolov8s(nc=2)
    model.train()
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        preds = model(x)
    return model, preds


class TestDetectionLoss:
    """Tests for the full DetectionLoss."""

    def test_detection_loss_no_detect_attr(self):
        """Model without .detect raises ValueError."""
        dummy = torch.nn.Linear(10, 10)
        with pytest.raises(ValueError, match="detect"):
            DetectionLoss(dummy)

    def test_detection_loss_backward(self, yolo_model_and_preds):
        """Full loss produces valid gradients on model parameters."""
        model, _ = yolo_model_and_preds
        model.train()
        model.zero_grad()
        x = torch.randn(1, 3, 320, 320)
        preds = model(x)

        loss_fn = DetectionLoss(model)
        # One target: [batch_idx, cls, x1, y1, x2, y2]
        targets = torch.tensor(
            [[0, 1, 50.0, 50.0, 200.0, 200.0]]
        )
        total_loss, loss_dict = loss_fn(preds, targets)
        assert torch.isfinite(total_loss)
        total_loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found on model params"

    def test_detection_loss_zero_targets(
        self, yolo_model_and_preds
    ):
        """Empty targets produce near-zero box/dfl loss."""
        model, preds = yolo_model_and_preds
        loss_fn = DetectionLoss(model)
        targets = torch.zeros(0, 6)
        total_loss, loss_dict = loss_fn(preds, targets)
        assert torch.isfinite(total_loss)
        # With no GT, box and dfl should be zero
        assert loss_dict["box"] == pytest.approx(0.0, abs=1e-6)
        assert loss_dict["dfl"] == pytest.approx(0.0, abs=1e-6)

    def test_detection_loss_single_target(
        self, yolo_model_and_preds
    ):
        """One target produces non-zero loss."""
        model, preds = yolo_model_and_preds
        loss_fn = DetectionLoss(model)
        targets = torch.tensor(
            [[0, 0, 80.0, 80.0, 240.0, 240.0]]
        )
        total_loss, loss_dict = loss_fn(preds, targets)
        assert torch.isfinite(total_loss)
        # cls loss should always be non-zero (BCE on scores)
        assert loss_dict["cls"] > 0

    def test_detection_loss_dict_keys(
        self, yolo_model_and_preds
    ):
        """loss_dict contains exactly 'box', 'cls', 'dfl'."""
        model, preds = yolo_model_and_preds
        loss_fn = DetectionLoss(model)
        targets = torch.tensor(
            [[0, 0, 50.0, 50.0, 150.0, 150.0]]
        )
        _, loss_dict = loss_fn(preds, targets)
        assert set(loss_dict.keys()) == {"box", "cls", "dfl"}

    def test_loss_gains_affect_magnitude(self):
        """Doubling box gain doubles the bbox component."""
        torch.manual_seed(123)
        model = yolov8s(nc=2)
        model.train()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            preds = model(x)

        targets = torch.tensor(
            [[0, 0, 80.0, 80.0, 200.0, 200.0]]
        )

        loss_fn_1x = DetectionLoss(model, box=7.5)
        loss_fn_2x = DetectionLoss(model, box=15.0)

        _, d1 = loss_fn_1x(preds, targets)
        _, d2 = loss_fn_2x(preds, targets)

        # box loss should scale proportionally (2x gain -> 2x)
        if d1["box"] > 1e-7:
            ratio = d2["box"] / d1["box"]
            assert ratio == pytest.approx(2.0, rel=1e-3)

    def test_detection_loss_multiple_targets(
        self, yolo_model_and_preds
    ):
        """Multiple targets across the same batch image."""
        model, preds = yolo_model_and_preds
        loss_fn = DetectionLoss(model)
        targets = torch.tensor([
            [0, 0, 30.0, 30.0, 100.0, 100.0],
            [0, 1, 150.0, 150.0, 280.0, 280.0],
        ])
        total_loss, loss_dict = loss_fn(preds, targets)
        assert torch.isfinite(total_loss)
        assert total_loss.item() > 0
