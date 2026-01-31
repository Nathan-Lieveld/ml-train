"""Tests for the detection evaluation module (NMS, AP, evaluate)."""

import numpy as np
import pytest
import torch

from src.detect.eval import compute_ap, evaluate, non_max_suppression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_preds(boxes_and_scores: list[list[float]], batch: int = 1) -> torch.Tensor:
    """Build a (B, N, 7) prediction tensor (4 xyxy + 3 class scores).

    *boxes_and_scores* is a list of [x1, y1, x2, y2, s0, s1, s2] rows.
    The same detections are replicated for every image in the batch.
    """
    t = torch.tensor(boxes_and_scores, dtype=torch.float32)  # (N, 7)
    return t.unsqueeze(0).expand(batch, -1, -1).clone()


class MockModel(torch.nn.Module):
    """Model that always returns a fixed prediction tensor."""

    def __init__(self, predictions: torch.Tensor):
        super().__init__()
        self.predictions = predictions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictions

    def eval(self) -> "MockModel":
        return self


# ---------------------------------------------------------------------------
# NMS tests
# ---------------------------------------------------------------------------


class TestNMS:
    def test_nms_filters_low_conf(self):
        """Predictions whose max class score is below *conf* are removed."""
        preds = _make_preds([
            [10, 10, 50, 50, 0.9, 0.1, 0.1],  # max=0.9 -- kept
            [60, 60, 90, 90, 0.01, 0.02, 0.01],  # max=0.02 -- removed
        ])
        results = non_max_suppression(preds, conf=0.5, iou=0.5)
        assert len(results) == 1
        det = results[0]
        assert det.shape[0] == 1
        assert det[0, 4].item() == pytest.approx(0.9)

    def test_nms_suppresses_overlapping(self):
        """Two same-class overlapping boxes: only the higher-conf one survives."""
        preds = _make_preds([
            [10, 10, 50, 50, 0.9, 0.1, 0.1],
            [12, 12, 48, 48, 0.7, 0.1, 0.1],  # heavily overlaps, same class
        ])
        results = non_max_suppression(preds, conf=0.05, iou=0.3)
        det = results[0]
        assert det.shape[0] == 1
        assert det[0, 4].item() == pytest.approx(0.9)

    def test_nms_preserves_different_classes(self):
        """Overlapping boxes of *different* classes should both survive."""
        preds = _make_preds([
            [10, 10, 50, 50, 0.9, 0.1, 0.1],  # class 0
            [12, 12, 48, 48, 0.1, 0.1, 0.8],  # class 2
        ])
        results = non_max_suppression(preds, conf=0.05, iou=0.3)
        det = results[0]
        assert det.shape[0] == 2
        classes = set(det[:, 5].tolist())
        assert classes == {0.0, 2.0}

    def test_nms_max_det(self):
        """Output is clipped to *max_det* detections."""
        rows = [
            [float(i * 60), 0.0, float(i * 60 + 50), 50.0, 0.9, 0.1, 0.1]
            for i in range(20)
        ]
        preds = _make_preds(rows)
        results = non_max_suppression(preds, conf=0.05, iou=0.5, max_det=5)
        assert results[0].shape[0] <= 5

    def test_nms_empty_input(self):
        """Zero predictions should yield an empty (0, 6) tensor."""
        preds = torch.zeros(1, 0, 7)
        results = non_max_suppression(preds, conf=0.5, iou=0.5)
        assert len(results) == 1
        assert results[0].shape == (0, 6)

    def test_nms_batch(self):
        """Multi-image batch produces one result tensor per image."""
        preds = _make_preds(
            [[10, 10, 50, 50, 0.9, 0.1, 0.1]],
            batch=4,
        )
        results = non_max_suppression(preds, conf=0.05, iou=0.5)
        assert len(results) == 4
        for det in results:
            assert det.shape[0] == 1
            assert det.shape[1] == 6

    def test_nms_rejects_non_3d(self):
        """Passing a 2-D tensor must raise ValueError."""
        with pytest.raises(ValueError, match="3D"):
            non_max_suppression(torch.zeros(10, 7))


# ---------------------------------------------------------------------------
# AP computation tests
# ---------------------------------------------------------------------------


class TestComputeAP:
    def test_ap_perfect(self):
        """Precision = 1 at every recall level gives AP ~ 1.0.

        The implementation appends a 0-precision sentinel at recall=1,
        so the interpolated value at the very last recall point drops
        slightly below 1.0.  We allow a small tolerance.
        """
        recall = np.linspace(0, 1, 50)
        precision = np.ones_like(recall)
        assert compute_ap(recall, precision) == pytest.approx(1.0, abs=0.02)

    def test_ap_zero(self):
        """Precision = 0 everywhere gives AP = 0.0."""
        recall = np.linspace(0, 1, 50)
        precision = np.zeros_like(recall)
        assert compute_ap(recall, precision) == pytest.approx(0.0, abs=1e-6)

    def test_ap_known_curve(self):
        """A hand-constructed PR curve should produce a predictable AP.

        Recall:    [0.0, 0.5, 1.0]
        Precision: [1.0, 0.5, 0.0]

        After monotonic-decreasing envelope and sentinel padding:
          mrec = [0.0, 0.0, 0.5, 1.0, 1.0]
          mpre = [1.0, 1.0, 0.5, 0.0, 0.0]

        101-point interpolation at x = linspace(0,1,101):
          - x in [0, 0.5):  interp -> ~1.0 down to 0.5
          - x in [0.5, 1.0]: interp -> 0.5 down to 0.0
        Expected AP ~ 0.5 (within reasonable tolerance).
        """
        recall = np.array([0.0, 0.5, 1.0])
        precision = np.array([1.0, 0.5, 0.0])
        ap = compute_ap(recall, precision)
        assert 0.3 < ap < 0.7  # approximate -- exact value depends on interp


# ---------------------------------------------------------------------------
# Full evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    @staticmethod
    def _make_dataloader(images, targets):
        """Return a single-batch dataloader (a plain list)."""
        return [{"images": images, "targets": targets}]

    def test_evaluate_perfect_predictions(self):
        """When model output matches GT exactly, mAP50 should be ~1.0."""
        device = torch.device("cpu")
        nc = 3
        # Ground-truth: one object, class 0, box [10, 10, 50, 50]
        # targets format: (N, 6) -> [batch_idx, cls, x1, y1, x2, y2]
        targets = torch.tensor([[0, 0, 10.0, 10.0, 50.0, 50.0]])

        # Model prediction: (1, 1, 7) -- matches GT box, high conf on class 0
        pred = torch.zeros(1, 1, 4 + nc)
        pred[0, 0, :4] = torch.tensor([10.0, 10.0, 50.0, 50.0])
        pred[0, 0, 4] = 0.95  # class 0 score
        pred[0, 0, 5] = 0.01  # class 1 score
        pred[0, 0, 6] = 0.01  # class 2 score

        model = MockModel(pred)
        images = torch.randn(1, 3, 64, 64)
        dl = self._make_dataloader(images, targets)

        result = evaluate(model, dl, device, conf=0.001, iou_nms=0.5)
        assert result["mAP50"] == pytest.approx(1.0, abs=0.05)

    def test_evaluate_no_predictions(self):
        """No ground-truth objects and empty model output -> mAP = 0."""
        device = torch.device("cpu")
        nc = 3
        # Empty targets: no ground-truth objects in this batch
        targets = torch.zeros(0, 6)

        # Model emits an empty prediction tensor (no candidate boxes)
        pred = torch.zeros(1, 0, 4 + nc)

        model = MockModel(pred)
        images = torch.randn(1, 3, 64, 64)
        dl = self._make_dataloader(images, targets)

        result = evaluate(model, dl, device, conf=0.001, iou_nms=0.5)
        assert result["mAP50"] == pytest.approx(0.0)
        assert result["mAP50_95"] == pytest.approx(0.0)

    def test_evaluate_wrong_class(self):
        """Correct box but wrong class -> mAP = 0."""
        device = torch.device("cpu")
        nc = 3
        # GT is class 0
        targets = torch.tensor([[0, 0, 10.0, 10.0, 50.0, 50.0]])

        # Prediction has high score only on class 2 (wrong class)
        pred = torch.zeros(1, 1, 4 + nc)
        pred[0, 0, :4] = torch.tensor([10.0, 10.0, 50.0, 50.0])
        pred[0, 0, 4] = 0.01  # class 0
        pred[0, 0, 5] = 0.01  # class 1
        pred[0, 0, 6] = 0.95  # class 2

        model = MockModel(pred)
        images = torch.randn(1, 3, 64, 64)
        dl = self._make_dataloader(images, targets)

        result = evaluate(model, dl, device, conf=0.001, iou_nms=0.5)
        assert result["mAP50"] == pytest.approx(0.0)

    def test_evaluate_returns_dict(self):
        """Output dict must contain both mAP50 and mAP50_95 keys."""
        device = torch.device("cpu")
        nc = 3
        targets = torch.tensor([[0, 0, 10.0, 10.0, 50.0, 50.0]])

        pred = torch.zeros(1, 1, 4 + nc)
        pred[0, 0, :4] = torch.tensor([10.0, 10.0, 50.0, 50.0])
        pred[0, 0, 4] = 0.9

        model = MockModel(pred)
        images = torch.randn(1, 3, 64, 64)
        dl = self._make_dataloader(images, targets)

        result = evaluate(model, dl, device, conf=0.001, iou_nms=0.5)
        assert isinstance(result, dict)
        assert "mAP50" in result
        assert "mAP50_95" in result
        assert isinstance(result["mAP50"], float)
        assert isinstance(result["mAP50_95"], float)
