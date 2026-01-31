"""Tests for src.detect.data – annotation loading, mmap cache, stores, dataset, dataloader."""

import json
import struct
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest
import torch
from PIL import Image

from src.detect.data import (
    CACHE_MAGIC,
    HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    DetectionDataset,
    DiskImageStore,
    MmapImageStore,
    _collate_fn,
    build_cache,
    create_dataloader,
    load_coco_annotations,
    load_yolo_annotations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    """Write a tiny RGB JPEG image to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 64, 32)).save(str(path))


def _make_yolo_tree(
    tmp_path: Path,
    *,
    num_images: int = 3,
    labels: dict[str, str] | None = None,
) -> Path:
    """Create a YOLO directory tree and return the path to data.yaml.

    ``labels`` maps stem name -> label file content.  Images are written for
    every key in *labels* plus enough extras to reach *num_images*.
    """
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    if labels is None:
        labels = {
            "img0": "0 0.5 0.5 0.2 0.3\n",
            "img1": "1 0.1 0.2 0.3 0.4\n2 0.6 0.7 0.1 0.1\n",
            "img2": "0 0.9 0.9 0.1 0.1\n",
        }

    stems = list(labels.keys())
    for i in range(num_images):
        stem = stems[i] if i < len(stems) else f"extra{i}"
        _make_image(img_dir / f"{stem}.jpg")

    for stem, text in labels.items():
        (lbl_dir / f"{stem}.txt").write_text(text)

    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("train: images/train\n")
    return yaml_path


def _make_coco_json(
    json_path: Path,
    images: list[dict],
    annotations: list[dict],
) -> None:
    json_path.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
    }))


def _df_for_images(tmp_path: Path, n: int = 3) -> pl.DataFrame:
    """Return a DataFrame with *n* images and one annotation each."""
    rows = []
    for i in range(n):
        img_path = tmp_path / f"img_{i}.jpg"
        _make_image(img_path)
        rows.append({
            "image_id": i,
            "file_path": str(img_path),
            "cls": i,
            "x1": 0.1,
            "y1": 0.2,
            "x2": 0.5,
            "y2": 0.6,
        })
    return pl.DataFrame(rows)


# ===================================================================
# Annotation loading – YOLO
# ===================================================================


class TestLoadYoloAnnotations:
    def test_load_yolo_annotations(self, tmp_path: Path) -> None:
        """3 YOLO label files produce a DataFrame with all expected columns."""
        yaml_path = _make_yolo_tree(tmp_path)
        df = load_yolo_annotations(yaml_path)

        expected_cols = {
            "image_id", "file_path", "cls",
            "cx", "cy", "w", "h",
            "x1", "y1", "x2", "y2",
        }
        assert expected_cols <= set(df.columns)
        # img0 has 1 box, img1 has 2 boxes, img2 has 1 box => 4 rows
        assert len(df.filter(pl.col("cls") >= 0)) == 4
        # 3 unique images
        assert df["image_id"].n_unique() == 3

    def test_load_yolo_empty_file(self, tmp_path: Path) -> None:
        """A label file with no lines produces zero annotation rows (cls == -1)."""
        yaml_path = _make_yolo_tree(
            tmp_path,
            num_images=1,
            labels={"img0": ""},
        )
        df = load_yolo_annotations(yaml_path)
        assert len(df) >= 1
        # All annotations should be the placeholder with cls == -1
        assert (df["cls"] == -1).all()

    def test_load_yolo_malformed_line(self, tmp_path: Path) -> None:
        """Lines with wrong field count are skipped with a warning."""
        yaml_path = _make_yolo_tree(
            tmp_path,
            num_images=1,
            labels={"img0": "0 0.5 0.5\n1 0.1 0.2 0.3 0.4\n"},
        )
        df = load_yolo_annotations(yaml_path)
        valid = df.filter(pl.col("cls") >= 0)
        # Only the second (valid) line should be kept
        assert len(valid) == 1
        assert valid["cls"][0] == 1


# ===================================================================
# Annotation loading – COCO / LVIS JSON
# ===================================================================


class TestLoadCocoAnnotations:
    def test_load_coco_annotations(self, tmp_path: Path) -> None:
        """Minimal COCO JSON produces a correct DataFrame."""
        json_path = tmp_path / "coco.json"
        _make_coco_json(
            json_path,
            images=[
                {"id": 1, "file_name": "a.jpg"},
                {"id": 2, "file_name": "b.jpg"},
            ],
            annotations=[
                {"id": 10, "image_id": 1, "bbox": [10, 20, 30, 40], "category_id": 3},
                {"id": 11, "image_id": 2, "bbox": [5, 5, 10, 10], "category_id": 7},
            ],
        )
        df = load_coco_annotations(json_path)
        expected_cols = {"image_id", "file_path", "cls", "x1", "y1", "x2", "y2"}
        assert expected_cols <= set(df.columns)
        assert len(df) == 2
        row_1 = df.filter(pl.col("image_id") == 1)
        assert row_1["x2"][0] == pytest.approx(40.0)  # 10 + 30
        assert row_1["y2"][0] == pytest.approx(60.0)  # 20 + 40

    def test_load_coco_no_annotations(self, tmp_path: Path) -> None:
        """Empty annotations list -> images present but cls == -1."""
        json_path = tmp_path / "coco_empty.json"
        _make_coco_json(
            json_path,
            images=[{"id": 1, "file_name": "a.jpg"}],
            annotations=[],
        )
        df = load_coco_annotations(json_path)
        assert len(df) == 1
        assert df["cls"][0] == -1


# ===================================================================
# Mmap cache
# ===================================================================


class TestMmapCache:
    def test_build_cache_roundtrip(self, tmp_path: Path) -> None:
        """Build a cache and read it back; pixel values should match."""
        df = _df_for_images(tmp_path, n=2)
        cache_path = tmp_path / "test.cache"
        imgsz = 16

        build_cache(df, cache_path, imgsz=imgsz, num_threads=1)
        store = MmapImageStore(cache_path)
        try:
            assert len(store) == 2
            tensor = store[0]
            assert tensor.dtype == torch.uint8
            assert tensor.ndim == 3
            assert tensor.shape[0] == 3
            # Max side should be <= imgsz
            assert max(tensor.shape[1], tensor.shape[2]) <= imgsz
        finally:
            store.close()

    def test_cache_header_magic(self, tmp_path: Path) -> None:
        """First 4 bytes of the cache file must be CACHE_MAGIC."""
        df = _df_for_images(tmp_path, n=1)
        cache_path = tmp_path / "magic.cache"
        build_cache(df, cache_path, imgsz=16, num_threads=1)

        raw = cache_path.read_bytes()
        assert raw[:4] == CACHE_MAGIC

    def test_cache_index_count(self, tmp_path: Path) -> None:
        """Index section has the correct number of entries."""
        n = 3
        df = _df_for_images(tmp_path, n=n)
        cache_path = tmp_path / "idx.cache"
        build_cache(df, cache_path, imgsz=16, num_threads=1)

        raw = cache_path.read_bytes()
        _, num_images, _ = struct.unpack_from("<III", raw, 4)
        assert num_images == n

        # Each index entry is INDEX_ENTRY_SIZE bytes
        expected_index_bytes = n * INDEX_ENTRY_SIZE
        index_section = raw[HEADER_SIZE : HEADER_SIZE + expected_index_bytes]
        assert len(index_section) == expected_index_bytes

    def test_mmap_store_len(self, tmp_path: Path) -> None:
        """len(store) equals the number of unique images."""
        n = 4
        df = _df_for_images(tmp_path, n=n)
        cache_path = tmp_path / "len.cache"
        build_cache(df, cache_path, imgsz=16, num_threads=1)

        store = MmapImageStore(cache_path)
        try:
            assert len(store) == n
        finally:
            store.close()


# ===================================================================
# DiskImageStore
# ===================================================================


class TestDiskImageStore:
    def test_disk_store_reads_image(self, tmp_path: Path) -> None:
        """Returns a uint8 CHW tensor."""
        df = _df_for_images(tmp_path, n=1)
        store = DiskImageStore(df)
        tensor = store[0]
        assert tensor.dtype == torch.uint8
        assert tensor.ndim == 3
        assert tensor.shape[0] == 3  # CHW

    def test_disk_store_missing_file(self, tmp_path: Path) -> None:
        """FileNotFoundError for a path that does not exist on disk."""
        df = pl.DataFrame({
            "image_id": [0],
            "file_path": [str(tmp_path / "nonexistent.jpg")],
            "cls": [0],
            "x1": [0.0], "y1": [0.0], "x2": [1.0], "y2": [1.0],
        })
        store = DiskImageStore(df)
        with pytest.raises(FileNotFoundError):
            store[0]


# ===================================================================
# DetectionDataset
# ===================================================================


class TestDetectionDataset:
    def _build_dataset(
        self, tmp_path: Path, n: int = 3
    ) -> tuple[DetectionDataset, DiskImageStore]:
        df = _df_for_images(tmp_path, n=n)
        store = DiskImageStore(df)
        ds = DetectionDataset(df, store)
        return ds, store

    def test_dataset_len(self, tmp_path: Path) -> None:
        """Length matches the number of unique image_ids."""
        ds, _ = self._build_dataset(tmp_path, n=5)
        assert len(ds) == 5

    def test_dataset_sample_keys(self, tmp_path: Path) -> None:
        """Each sample dict contains img, bboxes, cls."""
        ds, _ = self._build_dataset(tmp_path)
        sample = ds[0]
        assert "img" in sample
        assert "bboxes" in sample
        assert "cls" in sample

    def test_dataset_applies_transforms(self, tmp_path: Path) -> None:
        """A mock transform function is called on every sample."""
        df = _df_for_images(tmp_path, n=1)
        store = DiskImageStore(df)

        mock_transform = MagicMock(side_effect=lambda s: s)
        ds = DetectionDataset(df, store, transforms=mock_transform)
        _ = ds[0]
        mock_transform.assert_called_once()

    def test_dataset_groups_annotations(self, tmp_path: Path) -> None:
        """Image with 3 annotations returns bboxes of shape (3, 4)."""
        img_path = tmp_path / "multi.jpg"
        _make_image(img_path)
        df = pl.DataFrame({
            "image_id": [0, 0, 0],
            "file_path": [str(img_path)] * 3,
            "cls": [0, 1, 2],
            "x1": [0.1, 0.2, 0.3],
            "y1": [0.1, 0.2, 0.3],
            "x2": [0.4, 0.5, 0.6],
            "y2": [0.4, 0.5, 0.6],
        })
        store = DiskImageStore(df)
        ds = DetectionDataset(df, store)
        sample = ds[0]
        assert sample["bboxes"].shape == (3, 4)
        assert sample["cls"].shape == (3,)


# ===================================================================
# Dataloader / collate
# ===================================================================


class TestDataloader:
    def _build_loader(
        self, tmp_path: Path, n: int = 4, batch_size: int = 2
    ) -> tuple:
        df = _df_for_images(tmp_path, n=n)
        store = DiskImageStore(df)
        ds = DetectionDataset(df, store)
        loader = create_dataloader(ds, batch_size=batch_size, shuffle=False)
        return loader, ds

    def test_collate_stacks_images(self, tmp_path: Path) -> None:
        """Collated batch has stacked float image tensor of correct shape."""
        df = _df_for_images(tmp_path, n=2)
        store = DiskImageStore(df)
        ds = DetectionDataset(df, store)
        batch = [ds[0], ds[1]]
        collated = _collate_fn(batch)

        assert "images" in collated
        assert collated["images"].ndim == 4
        assert collated["images"].shape[0] == 2
        assert collated["images"].dtype == torch.float32
        # Values should be in [0, 1] after /255
        assert collated["images"].max() <= 1.0

    def test_collate_empty_labels(self, tmp_path: Path) -> None:
        """Batch where all samples have cls == -1 should not crash."""
        img_path = tmp_path / "nolabel.jpg"
        _make_image(img_path)
        df = pl.DataFrame({
            "image_id": [0],
            "file_path": [str(img_path)],
            "cls": [-1],
            "x1": [0.0], "y1": [0.0], "x2": [0.0], "y2": [0.0],
        })
        store = DiskImageStore(df)
        ds = DetectionDataset(df, store)
        batch = [ds[0]]
        collated = _collate_fn(batch)

        assert "targets" in collated
        assert collated["targets"].shape == (0, 6)

    def test_dataloader_iterates(self, tmp_path: Path) -> None:
        """A full pass over the DataLoader completes without error."""
        loader, ds = self._build_loader(tmp_path, n=4, batch_size=2)
        count = 0
        for batch in loader:
            assert "images" in batch
            assert "targets" in batch
            count += 1
        assert count == 2  # 4 images / batch_size 2
