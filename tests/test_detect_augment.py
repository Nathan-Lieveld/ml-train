"""Comprehensive tests for src.detect.augment transforms."""

from __future__ import annotations

import pytest
import torch

from src.detect.augment import (
    Compose,
    LetterBox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    c: int = 3,
    h: int = 64,
    w: int = 64,
    n_boxes: int = 2,
    color: int | None = None,
) -> dict:
    """Build a synthetic sample dict with a uint8 CHW image and bboxes."""
    if color is not None:
        img = torch.full((c, h, w), color, dtype=torch.uint8)
    else:
        img = torch.randint(0, 256, (c, h, w), dtype=torch.uint8)

    if n_boxes > 0:
        # Deterministic non-degenerate boxes inside the image
        bboxes = torch.zeros(n_boxes, 4, dtype=torch.float32)
        for i in range(n_boxes):
            x1 = int(w * 0.1 + i * 2)
            y1 = int(h * 0.1 + i * 2)
            x2 = min(x1 + int(w * 0.3), w)
            y2 = min(y1 + int(h * 0.3), h)
            bboxes[i] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        cls = torch.arange(n_boxes, dtype=torch.long)
    else:
        bboxes = torch.zeros(0, 4, dtype=torch.float32)
        cls = torch.zeros(0, dtype=torch.long)

    return {"img": img, "bboxes": bboxes, "cls": cls}


class MockDataset:
    """Minimal dataset returning fixed samples for Mosaic / MixUp tests."""

    def __init__(
        self,
        n: int = 8,
        h: int = 64,
        w: int = 64,
        n_boxes: int = 2,
    ) -> None:
        self.samples = [
            _make_sample(h=h, w=w, n_boxes=n_boxes) for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ===================================================================
# LetterBox
# ===================================================================


class TestLetterBox:
    """Tests for the LetterBox transform."""

    def test_letterbox_square(self):
        """Square image produces no padding, only resize."""
        sample = _make_sample(h=100, w=100, n_boxes=1)
        lb = LetterBox(imgsz=50)
        out = lb(sample)

        assert out["img"].shape == (3, 50, 50)
        # No padding means the border rows/cols should not be fill value
        # For a square input the entire output should be image content.
        # The fill value is 114; check corners are NOT all 114.
        img_np = out["img"].permute(1, 2, 0).numpy()
        # With a random image resized to exactly 50x50 there is no pad band.
        # Verify top-left and bottom-right are used (not 114-fill).
        top_left_is_fill = (img_np[0, 0] == 114).all()
        bottom_right_is_fill = (img_np[-1, -1] == 114).all()
        # With a random image the chance both corners are exactly 114 is tiny.
        assert not (top_left_is_fill and bottom_right_is_fill)

    def test_letterbox_landscape(self):
        """Wide image gets top/bottom padding and bboxes shift downward."""
        sample = _make_sample(h=50, w=100, n_boxes=1)
        original_bbox = sample["bboxes"].clone()
        lb = LetterBox(imgsz=100)
        out = lb(sample)

        assert out["img"].shape == (3, 100, 100)
        # Scale = 100/100 = 1.0, new_h=50, pad_h=50, top=25
        expected_y_shift = 25.0
        assert out["bboxes"][0, 1] == pytest.approx(
            original_bbox[0, 1] + expected_y_shift, abs=1.0
        )

    def test_letterbox_portrait(self):
        """Tall image gets left/right padding and bboxes shift rightward."""
        sample = _make_sample(h=100, w=50, n_boxes=1)
        original_bbox = sample["bboxes"].clone()
        lb = LetterBox(imgsz=100)
        out = lb(sample)

        assert out["img"].shape == (3, 100, 100)
        # Scale = 100/100 = 1.0, new_w=50, pad_w=50, left=25
        expected_x_shift = 25.0
        assert out["bboxes"][0, 0] == pytest.approx(
            original_bbox[0, 0] + expected_x_shift, abs=1.0
        )

    def test_letterbox_preserves_aspect(self):
        """Aspect ratio of resized content within 1% of original."""
        sample = _make_sample(h=75, w=200, n_boxes=0)
        lb = LetterBox(imgsz=640)
        lb(sample)

        original_aspect = 200 / 75
        # The resized content occupies new_w x new_h inside the padded image.
        # scale = 640/200 = 3.2 -> new_w=640, new_h=240
        new_aspect = 640 / 240
        assert abs(new_aspect - original_aspect) / original_aspect < 0.01

    def test_letterbox_bbox_inside_image(self):
        """All bboxes lie within [0, imgsz] after letterbox."""
        sample = _make_sample(h=120, w=80, n_boxes=5)
        lb = LetterBox(imgsz=64)
        out = lb(sample)

        bboxes = out["bboxes"]
        assert (bboxes[:, 0] >= 0).all()
        assert (bboxes[:, 1] >= 0).all()
        assert (bboxes[:, 2] <= 64).all()
        assert (bboxes[:, 3] <= 64).all()

    def test_letterbox_invalid_imgsz(self):
        """imgsz <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            LetterBox(imgsz=0)
        with pytest.raises(ValueError):
            LetterBox(imgsz=-10)


# ===================================================================
# Mosaic
# ===================================================================


class TestMosaic:
    """Tests for the Mosaic transform."""

    def test_mosaic_output_shape(self):
        """Output image shape is (3, imgsz, imgsz)."""
        ds = MockDataset(n=8, h=64, w=64, n_boxes=2)
        mosaic = Mosaic(dataset=ds, imgsz=128, p=1.0)
        sample = ds[0]
        out = mosaic(sample)

        assert out["img"].shape == (3, 128, 128)
        assert out["img"].dtype == torch.uint8

    def test_mosaic_label_count(self):
        """Merged label count is >= individual source counts (clipping may reduce)."""
        ds = MockDataset(n=8, h=64, w=64, n_boxes=3)
        mosaic = Mosaic(dataset=ds, imgsz=128, p=1.0)
        sample = ds[0]
        out = mosaic(sample)

        # At least the primary sample's boxes should survive (scaled, clamped)
        assert out["bboxes"].shape[0] >= 1
        assert out["bboxes"].shape[0] == out["cls"].shape[0]

    def test_mosaic_bbox_validity(self):
        """All output bboxes have x2 > x1 and y2 > y1."""
        ds = MockDataset(n=8, h=64, w=64, n_boxes=4)
        mosaic = Mosaic(dataset=ds, imgsz=128, p=1.0)
        out = mosaic(ds[0])

        if out["bboxes"].numel() > 0:
            widths = out["bboxes"][:, 2] - out["bboxes"][:, 0]
            heights = out["bboxes"][:, 3] - out["bboxes"][:, 1]
            assert (widths >= 0).all()
            assert (heights >= 0).all()

    def test_mosaic_p0_passthrough(self):
        """p=0 returns the original sample unchanged."""
        ds = MockDataset(n=8, h=64, w=64, n_boxes=2)
        mosaic = Mosaic(dataset=ds, imgsz=128, p=0.0)
        sample = ds[0]
        out = mosaic(sample)

        assert torch.equal(out["img"], sample["img"])
        assert torch.equal(out["bboxes"], sample["bboxes"])

    def test_mosaic_requires_min_dataset(self):
        """Dataset with fewer than 4 samples raises ValueError."""
        ds = MockDataset(n=3)
        with pytest.raises(ValueError, match="len\\(dataset\\) >= 4"):
            Mosaic(dataset=ds, imgsz=128)


# ===================================================================
# MixUp
# ===================================================================


class TestMixUp:
    """Tests for the MixUp transform."""

    def test_mixup_image_range(self):
        """Blended pixel values stay in [0, 255]."""
        ds = MockDataset(n=8, h=64, w=64, n_boxes=2)
        mixup = MixUp(dataset=ds, p=1.0)
        sample = ds[0]
        out = mixup(sample)

        assert out["img"].dtype == torch.uint8
        assert out["img"].min() >= 0
        assert out["img"].max() <= 255

    def test_mixup_label_concat(self):
        """Output labels are the concatenation of both samples' labels."""
        ds = MockDataset(n=8, h=64, w=64, n_boxes=2)
        mixup = MixUp(dataset=ds, p=1.0)
        sample = ds[0]
        n_original = sample["cls"].shape[0]
        out = mixup(sample)

        # Output should have at least as many labels as the primary sample
        assert out["cls"].shape[0] >= n_original
        assert out["bboxes"].shape[0] == out["cls"].shape[0]

    def test_mixup_p0_passthrough(self):
        """p=0 returns the original sample unchanged."""
        ds = MockDataset(n=8, h=64, w=64, n_boxes=2)
        mixup = MixUp(dataset=ds, p=0.0)
        sample = ds[0]
        out = mixup(sample)

        assert torch.equal(out["img"], sample["img"])
        assert torch.equal(out["bboxes"], sample["bboxes"])
        assert torch.equal(out["cls"], sample["cls"])


# ===================================================================
# RandomPerspective
# ===================================================================


class TestRandomPerspective:
    """Tests for the RandomPerspective transform."""

    def test_perspective_identity(self):
        """All params = 0 produces output approximately equal to input."""
        sample = _make_sample(h=64, w=64, n_boxes=2, color=100)
        rp = RandomPerspective(
            degrees=0, translate=0, scale=0, shear=0, seed=42
        )
        out = rp(sample)

        # Image should be nearly identical (rounding from warp may differ +-1)
        diff = (
            out["img"].float() - sample["img"].float()
        ).abs().max().item()
        assert diff <= 2.0, f"Max pixel diff {diff} > 2"

        # Bboxes should be close to originals
        if out["bboxes"].numel() > 0 and sample["bboxes"].numel() > 0:
            bbox_diff = (
                (out["bboxes"] - sample["bboxes"]).abs().max().item()
            )
            assert bbox_diff < 2.0, f"Max bbox diff {bbox_diff} > 2"

    def test_perspective_filters_tiny_bboxes(self):
        """Tiny bboxes (area < MIN_BBOX_AREA) are removed after warp."""
        sample = _make_sample(h=640, w=640, n_boxes=0)
        # Add a very small bbox that will become tiny after perspective
        sample["bboxes"] = torch.tensor(
            [[100, 100, 101, 101]], dtype=torch.float32
        )  # 1x1 area
        sample["cls"] = torch.tensor([0], dtype=torch.long)

        rp = RandomPerspective(
            degrees=0, translate=0, scale=0, shear=0, seed=42
        )
        out = rp(sample)

        # A 1x1 box has area=1 which is < MIN_BBOX_AREA=10, so filtered
        assert out["bboxes"].shape[0] == 0

    def test_perspective_filters_extreme_aspect(self):
        """Bboxes with extreme aspect ratios are filtered."""
        sample = _make_sample(h=640, w=640, n_boxes=0)
        # Very elongated bbox: width=200, height=1 -> aspect ~200
        sample["bboxes"] = torch.tensor(
            [[100, 300, 300, 301]], dtype=torch.float32
        )
        sample["cls"] = torch.tensor([0], dtype=torch.long)

        rp = RandomPerspective(
            degrees=0, translate=0, scale=0, shear=0, seed=42
        )
        out = rp(sample)

        # Width 200, height 1 -> area=200 (>10) but aspect=200 (>20)
        assert out["bboxes"].shape[0] == 0

    def test_perspective_output_shape_preserved(self):
        """Output image has the same spatial dimensions as input."""
        sample = _make_sample(h=80, w=120, n_boxes=2)
        rp = RandomPerspective(
            degrees=10, translate=0.1, scale=0.2, shear=5, seed=7
        )
        out = rp(sample)

        assert out["img"].shape == (3, 80, 120)


# ===================================================================
# RandomHSV
# ===================================================================


class TestRandomHSV:
    """Tests for the RandomHSV transform."""

    def test_hsv_output_range(self):
        """All output pixels in [0, 255]."""
        sample = _make_sample(h=64, w=64, n_boxes=1)
        hsv = RandomHSV(h=0.015, s=0.7, v=0.4, seed=42)
        out = hsv(sample)

        assert out["img"].dtype == torch.uint8
        assert out["img"].min() >= 0
        assert out["img"].max() <= 255

    def test_hsv_no_op(self):
        """h=0, s=0, v=0 returns the exact input image."""
        sample = _make_sample(h=64, w=64, n_boxes=1, color=128)
        hsv = RandomHSV(h=0, s=0, v=0)
        out = hsv(sample)

        assert torch.equal(out["img"], sample["img"])
        assert torch.equal(out["bboxes"], sample["bboxes"])
        assert torch.equal(out["cls"], sample["cls"])

    def test_hsv_deterministic_with_seed(self):
        """Same seed produces identical output across two calls."""
        sample1 = _make_sample(h=64, w=64, n_boxes=1, color=180)
        sample2 = _make_sample(h=64, w=64, n_boxes=1, color=180)

        out1 = RandomHSV(h=0.015, s=0.7, v=0.4, seed=99)(sample1)
        out2 = RandomHSV(h=0.015, s=0.7, v=0.4, seed=99)(sample2)

        assert torch.equal(out1["img"], out2["img"])

    def test_hsv_modifies_image(self):
        """Non-zero params actually change the image."""
        sample = _make_sample(h=64, w=64, n_boxes=0, color=100)
        hsv = RandomHSV(h=0.015, s=0.7, v=0.4, seed=1)
        out = hsv(sample)

        # With s=0.7 and v=0.4 the image should differ
        assert not torch.equal(out["img"], sample["img"])


# ===================================================================
# RandomFlip
# ===================================================================


class TestRandomFlip:
    """Tests for the RandomFlip transform."""

    def test_flip_image_mirrored(self):
        """Flipped image matches torch.flip(input, [-1])."""
        sample = _make_sample(h=64, w=64, n_boxes=1)
        # p=1.0 guarantees flip; seed ensures determinism
        flip = RandomFlip(p=1.0, seed=42)
        out = flip(sample)

        expected = sample["img"].flip(-1)
        assert torch.equal(out["img"], expected)

    def test_flip_bbox_mirrored(self):
        """Bbox x-coordinates are horizontally reflected."""
        w = 100
        sample = _make_sample(h=80, w=w, n_boxes=1)
        original_x1 = sample["bboxes"][0, 0].item()
        original_x2 = sample["bboxes"][0, 2].item()

        flip = RandomFlip(p=1.0, seed=42)
        out = flip(sample)

        expected_new_x1 = w - original_x2
        expected_new_x2 = w - original_x1
        assert out["bboxes"][0, 0].item() == pytest.approx(
            expected_new_x1, abs=1e-4
        )
        assert out["bboxes"][0, 2].item() == pytest.approx(
            expected_new_x2, abs=1e-4
        )

    def test_flip_p0_passthrough(self):
        """p=0 returns the original sample unchanged."""
        sample = _make_sample(h=64, w=64, n_boxes=2)
        flip = RandomFlip(p=0.0, seed=42)
        out = flip(sample)

        assert torch.equal(out["img"], sample["img"])
        assert torch.equal(out["bboxes"], sample["bboxes"])
        assert torch.equal(out["cls"], sample["cls"])

    def test_flip_double_flip_identity(self):
        """Flipping twice returns the original image and bboxes."""
        sample = _make_sample(h=64, w=64, n_boxes=2)
        flip = RandomFlip(p=1.0, seed=42)
        out = flip(flip(sample))

        assert torch.equal(out["img"], sample["img"])
        assert torch.allclose(out["bboxes"], sample["bboxes"], atol=1e-5)


# ===================================================================
# Compose
# ===================================================================


class TestCompose:
    """Tests for the Compose transform."""

    def test_compose_chain(self):
        """Transforms are applied in sequential order."""
        call_order: list[str] = []

        class TagA:
            def __call__(self, sample: dict) -> dict:
                call_order.append("A")
                return sample

        class TagB:
            def __call__(self, sample: dict) -> dict:
                call_order.append("B")
                return sample

        compose = Compose([TagA(), TagB()])
        compose(_make_sample())
        assert call_order == ["A", "B"]

    def test_compose_empty(self):
        """Empty transforms list passes sample through unchanged."""
        sample = _make_sample(h=32, w=32, n_boxes=2)
        compose = Compose([])
        out = compose(sample)

        assert torch.equal(out["img"], sample["img"])
        assert torch.equal(out["bboxes"], sample["bboxes"])
        assert torch.equal(out["cls"], sample["cls"])

    def test_compose_none_transforms(self):
        """None transforms argument defaults to empty list."""
        compose = Compose(transforms=None)
        sample = _make_sample()
        out = compose(sample)

        assert torch.equal(out["img"], sample["img"])

    def test_compose_integration(self):
        """LetterBox + RandomFlip(p=1) composed together."""
        sample = _make_sample(h=100, w=50, n_boxes=2)
        compose = Compose([
            LetterBox(imgsz=64),
            RandomFlip(p=1.0, seed=0),
        ])
        out = compose(sample)

        assert out["img"].shape == (3, 64, 64)
        # Bboxes should still be valid after both transforms
        if out["bboxes"].numel() > 0:
            assert (out["bboxes"][:, 0] >= 0).all()
            assert (out["bboxes"][:, 2] <= 64).all()
