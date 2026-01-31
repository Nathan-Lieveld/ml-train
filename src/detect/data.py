"""Detection data loading: annotation parsing, mmap image cache, Dataset, DataLoader."""

from __future__ import annotations

import json
import logging
import mmap
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torchvision.io
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_DF_COLUMNS = {"image_id", "file_path", "cls", "x1", "y1", "x2", "y2"}

# Mmap cache format constants
CACHE_MAGIC = b"YOLC"
CACHE_VERSION = 1
HEADER_SIZE = 16  # magic(4) + version(4) + num_images(4) + imgsz(4)
INDEX_ENTRY_SIZE = 16  # offset(8) + nbytes(4) + H(2) + W(2)


# ---------------------------------------------------------------------------
# Annotation loading — YOLO format
# ---------------------------------------------------------------------------


def load_yolo_annotations(data_yaml: str | Path) -> pl.DataFrame:
    """Parse YOLO-format labels directory from a data YAML file.

    The YAML must contain ``train`` and/or ``val`` keys pointing to image directories.
    Label files are expected next to the image directory under ``../labels/``.

    Args:
        data_yaml: Path to a YOLO-format ``.yaml`` file.

    Returns:
        DataFrame with columns: image_id, file_path, cls, cx, cy, w, h, x1, y1, x2, y2.
    """
    import yaml

    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
    if data_yaml.suffix not in (".yaml", ".yml"):
        raise ValueError(f"Expected .yaml file, got {data_yaml.suffix}")

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    root = data_yaml.parent
    rows: list[dict[str, Any]] = []
    image_id = 0

    for split in ("train", "val"):
        if split not in cfg:
            continue
        img_dir = root / cfg[split]
        if not img_dir.is_dir():
            logger.warning("Image directory not found: %s", img_dir)
            continue

        label_dir = img_dir.parent / "labels"
        if not label_dir.is_dir():
            label_dir = img_dir.parent.parent / "labels" / img_dir.name
        if not label_dir.is_dir():
            logger.warning("Label directory not found for %s", img_dir)
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.is_file():
                rows.append({
                    "image_id": image_id,
                    "file_path": str(img_path),
                    "cls": -1, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0,
                    "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0,
                })
                image_id += 1
                continue

            text = label_path.read_text()
            has_label = False
            for line_num, line in enumerate(text.splitlines()):
                parts = line.strip().split()
                if len(parts) != 5:
                    if parts:
                        logger.warning(
                            "Skipping malformed line %d in %s: expected 5 fields, got %d",
                            line_num, label_path, len(parts),
                        )
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    logger.warning("Skipping non-numeric line %d in %s", line_num, label_path)
                    continue

                rows.append({
                    "image_id": image_id,
                    "file_path": str(img_path),
                    "cls": cls_id,
                    "cx": cx, "cy": cy, "w": bw, "h": bh,
                    "x1": cx - bw / 2, "y1": cy - bh / 2,
                    "x2": cx + bw / 2, "y2": cy + bh / 2,
                })
                has_label = True

            if not has_label:
                rows.append({
                    "image_id": image_id,
                    "file_path": str(img_path),
                    "cls": -1, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0,
                    "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0,
                })
            image_id += 1

    if not rows:
        return pl.DataFrame(schema={
            "image_id": pl.Int64, "file_path": pl.Utf8, "cls": pl.Int64,
            "cx": pl.Float64, "cy": pl.Float64, "w": pl.Float64, "h": pl.Float64,
            "x1": pl.Float64, "y1": pl.Float64, "x2": pl.Float64, "y2": pl.Float64,
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Annotation loading — COCO / LVIS JSON
# ---------------------------------------------------------------------------


def load_coco_annotations(json_path: str | Path) -> pl.DataFrame:
    """Parse COCO/LVIS-format JSON annotations.

    Args:
        json_path: Path to a COCO-format ``.json`` file.

    Returns:
        DataFrame with columns: image_id, file_path, cls, x1, y1, x2, y2.
    """
    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if json_path.suffix != ".json":
        raise ValueError(f"Expected .json file, got {json_path.suffix}")

    with open(json_path) as f:
        data = json.load(f)

    if "images" not in data or "annotations" not in data:
        raise ValueError("JSON must have 'images' and 'annotations' keys")

    # Build image_id → file_name map
    id_to_file: dict[int, str] = {}
    for img_info in data["images"]:
        id_to_file[img_info["id"]] = img_info.get("file_name", "")

    rows: list[dict[str, Any]] = []
    annotated_ids: set[int] = set()

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        annotated_ids.add(img_id)
        x, y, bw, bh = ann["bbox"]  # COCO format: [x, y, w, h]
        cls_id = ann.get("category_id", 0)

        rows.append({
            "image_id": img_id,
            "file_path": id_to_file.get(img_id, ""),
            "cls": cls_id,
            "x1": x, "y1": y, "x2": x + bw, "y2": y + bh,
        })

    # Add images with no annotations
    for img_info in data["images"]:
        if img_info["id"] not in annotated_ids:
            rows.append({
                "image_id": img_info["id"],
                "file_path": img_info.get("file_name", ""),
                "cls": -1, "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0,
            })

    if not rows:
        return pl.DataFrame(schema={
            "image_id": pl.Int64, "file_path": pl.Utf8, "cls": pl.Int64,
            "x1": pl.Float64, "y1": pl.Float64, "x2": pl.Float64, "y2": pl.Float64,
        })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mmap image cache
# ---------------------------------------------------------------------------


def _load_and_resize(file_path: str, imgsz: int) -> tuple[np.ndarray, int, int]:
    """Load image, resize to max_side=imgsz (no padding), return CHW uint8 array."""
    img = torchvision.io.decode_image(
        torchvision.io.read_file(file_path),
        mode=torchvision.io.ImageReadMode.RGB,
    )  # (3, H, W)
    _, h, w = img.shape
    scale = imgsz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = torch.nn.functional.interpolate(
        img.unsqueeze(0).float(), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0).to(torch.uint8)
    return img.numpy(), new_h, new_w


def build_cache(
    df: pl.DataFrame,
    cache_path: str | Path,
    imgsz: int,
    num_threads: int = 8,
) -> None:
    """Build mmap image cache from a DataFrame of image paths.

    Binary format::

        [magic:4B][version:4B][num_images:4B][imgsz:4B]
        [offset:8B, nbytes:4B, H:2B, W:2B] * num_images   (index)
        [packed uint8 CHW tensors]                          (data)

    Args:
        df: DataFrame with ``image_id`` and ``file_path`` columns.
        cache_path: Output cache file path.
        imgsz: Target max side length.
        num_threads: Thread pool size for image loading.
    """
    cache_path = Path(cache_path)
    if imgsz <= 0:
        raise ValueError(f"imgsz must be > 0, got {imgsz}")
    if df.is_empty():
        raise ValueError("DataFrame is empty — nothing to cache")
    if not cache_path.parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {cache_path.parent}")

    unique_images = df.select("image_id", "file_path").unique(subset=["image_id"]).sort("image_id")
    num_images = len(unique_images)
    file_paths = unique_images["file_path"].to_list()

    logger.info("Building cache for %d images at %s", num_images, cache_path)

    # Load all images in parallel
    def _load(fp: str) -> tuple[np.ndarray, int, int]:
        return _load_and_resize(fp, imgsz)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        results = list(pool.map(_load, file_paths))

    # Write cache file
    with open(cache_path, "wb") as f:
        # Header
        f.write(CACHE_MAGIC)
        f.write(struct.pack("<III", CACHE_VERSION, num_images, imgsz))

        # Compute offsets for index
        data_start = HEADER_SIZE + INDEX_ENTRY_SIZE * num_images
        offset = data_start

        # Write index
        index_entries: list[tuple[int, int, int, int]] = []
        for arr, h, w in results:
            nbytes = arr.nbytes
            index_entries.append((offset, nbytes, h, w))
            offset += nbytes

        for off, nb, h, w in index_entries:
            f.write(struct.pack("<QI", off, nb))
            f.write(struct.pack("<HH", h, w))

        # Write data
        for arr, _, _ in results:
            f.write(arr.tobytes())

    logger.info("Cache written: %d images, %d bytes", num_images, offset)


class MmapImageStore:
    """Zero-copy image reads from an mmap cache file."""

    def __init__(self, cache_path: str | Path) -> None:
        cache_path = Path(cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        self._file = open(cache_path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # Read header
        magic = self._mm[:4]
        if magic != CACHE_MAGIC:
            raise ValueError(f"Invalid cache magic: {magic!r}")

        version, self._num_images, self._imgsz = struct.unpack_from("<III", self._mm, 4)
        if version != CACHE_VERSION:
            raise ValueError(f"Unsupported cache version: {version}")

        # Read index
        self._index: list[tuple[int, int, int, int]] = []
        for i in range(self._num_images):
            idx_offset = HEADER_SIZE + i * INDEX_ENTRY_SIZE
            offset, nbytes = struct.unpack_from("<QI", self._mm, idx_offset)
            h, w = struct.unpack_from("<HH", self._mm, idx_offset + 12)
            self._index.append((offset, nbytes, h, w))

    def __len__(self) -> int:
        return self._num_images

    def __getitem__(self, idx: int) -> torch.Tensor:
        offset, nbytes, h, w = self._index[idx]
        buf = self._mm[offset : offset + nbytes]
        np.frombuffer(buf, dtype=np.uint8).reshape(3, h, w)
        return torch.frombuffer(bytearray(buf), dtype=torch.uint8).reshape(3, h, w)

    def close(self) -> None:
        self._mm.close()
        self._file.close()

    @property
    def imgsz(self) -> int:
        return self._imgsz


# ---------------------------------------------------------------------------
# Disk image store (fallback, no cache)
# ---------------------------------------------------------------------------


class DiskImageStore:
    """Read images from disk on every access (no caching)."""

    def __init__(self, df: pl.DataFrame) -> None:
        unique = df.select("image_id", "file_path").unique(subset=["image_id"]).sort("image_id")
        self._paths = unique["file_path"].to_list()
        self._image_ids = unique["image_id"].to_list()

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        fp = self._paths[idx]
        path = Path(fp)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {fp}")
        return torchvision.io.decode_image(
            torchvision.io.read_file(fp),
            mode=torchvision.io.ImageReadMode.RGB,
        )


# ---------------------------------------------------------------------------
# Detection dataset
# ---------------------------------------------------------------------------


class DetectionDataset(Dataset[dict[str, Any]]):
    """Detection dataset: groups annotations by image_id, returns sample dicts."""

    def __init__(
        self,
        df: pl.DataFrame,
        image_store: MmapImageStore | DiskImageStore,
        transforms: Any | None = None,
        imgsz: int = 640,
    ) -> None:
        if len(image_store) == 0:
            raise ValueError("image_store must not be empty")
        missing = REQUIRED_DF_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self.df = df
        self.store = image_store
        self.transforms = transforms
        self.imgsz = imgsz

        # Group annotations by image_id
        self._image_ids = sorted(df["image_id"].unique().to_list())
        self._annotations: dict[int, pl.DataFrame] = {}
        for img_id in self._image_ids:
            self._annotations[img_id] = df.filter(pl.col("image_id") == img_id)

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_id = self._image_ids[idx]
        img = self.store[idx]  # (3, H, W) uint8

        ann = self._annotations[img_id]
        valid = ann.filter(pl.col("cls") >= 0)

        if len(valid) > 0:
            bboxes = torch.tensor(
                valid.select("x1", "y1", "x2", "y2").to_numpy(), dtype=torch.float32
            )
            cls = torch.tensor(valid["cls"].to_list(), dtype=torch.long)
        else:
            bboxes = torch.zeros(0, 4, dtype=torch.float32)
            cls = torch.zeros(0, dtype=torch.long)

        sample: dict[str, Any] = {"img": img, "bboxes": bboxes, "cls": cls}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


# ---------------------------------------------------------------------------
# DataLoader with custom collate
# ---------------------------------------------------------------------------


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate detection samples: stack images, pad labels per image."""
    images = torch.stack([s["img"] for s in batch]).float() / 255.0

    # Build flat targets: [batch_idx, cls, x1, y1, x2, y2]
    targets_list: list[torch.Tensor] = []
    for i, s in enumerate(batch):
        bboxes = s["bboxes"]
        cls = s["cls"]
        if bboxes.numel() > 0:
            n = bboxes.shape[0]
            batch_idx = torch.full((n, 1), i, dtype=torch.float32)
            cls_col = cls.float().unsqueeze(1) if cls.dim() == 1 else cls.float()
            targets_list.append(torch.cat([batch_idx, cls_col, bboxes.float()], dim=1))

    if targets_list:
        targets = torch.cat(targets_list, 0)
    else:
        targets = torch.zeros(0, 6)

    return {"images": images, "targets": targets}


def create_dataloader(
    dataset: DetectionDataset,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader[dict[str, Any]]:
    """Create a detection DataLoader with custom collation.

    Uses num_workers=0 for cross-platform safety.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate_fn,
        drop_last=False,
    )
