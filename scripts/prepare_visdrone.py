"""Download VisDrone2019-DET and convert annotations to YOLO format.

VisDrone annotation format (per line, comma-separated):
    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion

Categories kept (remapped to 0-9):
    1:pedestrian, 2:people, 3:bicycle, 4:car, 5:van,
    6:truck, 7:tricycle, 8:awning-tricycle, 9:bus, 10:motor

Categories 0 (ignored regions) and 11 (others) are skipped.
"""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import yaml
from PIL import Image

VISDRONE_URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
}

CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# VisDrone category IDs 1-10 → YOLO class IDs 0-9
CATEGORY_REMAP = {i: i - 1 for i in range(1, 11)}


def download_and_extract(url: str, dest: Path) -> Path:
    """Download a zip and extract it. Returns the extracted directory."""
    zip_path = dest / url.rsplit("/", 1)[-1]
    if not zip_path.is_file():
        print(f"Downloading {url} ...")
        urlretrieve(url, zip_path)
        print(f"  Saved to {zip_path}")
    else:
        print(f"  Already downloaded: {zip_path}")

    # Find the root folder inside the zip
    with zipfile.ZipFile(zip_path) as zf:
        top_dirs = {n.split("/")[0] for n in zf.namelist() if "/" in n}
        extract_name = top_dirs.pop() if len(top_dirs) == 1 else zip_path.stem
        extracted = dest / extract_name
        if not extracted.is_dir():
            print(f"  Extracting to {extracted} ...")
            zf.extractall(dest)
    return extracted


def convert_split(src_dir: Path, img_dst: Path, lbl_dst: Path) -> int:
    """Convert a VisDrone split to YOLO format. Returns number of images processed."""
    img_src = src_dir / "images"
    ann_src = src_dir / "annotations"
    if not img_src.is_dir() or not ann_src.is_dir():
        raise FileNotFoundError(f"Expected images/ and annotations/ in {src_dir}")

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for ann_path in sorted(ann_src.glob("*.txt")):
        stem = ann_path.stem
        # Find matching image
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = img_src / (stem + ext)
            if candidate.is_file():
                img_path = candidate
                break
        if img_path is None:
            continue

        # Get image dimensions
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        # Convert annotations
        lines: list[str] = []
        for raw_line in ann_path.read_text().splitlines():
            parts = raw_line.strip().split(",")
            if len(parts) < 8:
                continue
            bbox_left, bbox_top, bbox_w, bbox_h = (
                int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]),
            )
            category = int(parts[5])

            if category not in CATEGORY_REMAP:
                continue  # skip ignored/other
            if bbox_w <= 0 or bbox_h <= 0:
                continue

            cls_id = CATEGORY_REMAP[category]
            cx = (bbox_left + bbox_w / 2) / img_w
            cy = (bbox_top + bbox_h / 2) / img_h
            w = bbox_w / img_w
            h = bbox_h / img_h

            # Clamp to [0, 1]
            cx, cy, w, h = (
                max(0.0, min(1.0, cx)),
                max(0.0, min(1.0, cy)),
                max(0.0, min(1.0, w)),
                max(0.0, min(1.0, h)),
            )
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Copy image and write label
        shutil.copy2(img_path, img_dst / img_path.name)
        (lbl_dst / (stem + ".txt")).write_text("\n".join(lines) + "\n" if lines else "")
        count += 1

    return count


def main() -> None:
    data_root = Path(__file__).resolve().parent.parent / "data" / "visdrone"
    data_root.mkdir(parents=True, exist_ok=True)

    for split, url in VISDRONE_URLS.items():
        print(f"\n--- {split} split ---")
        extracted = download_and_extract(url, data_root)
        n = convert_split(
            extracted,
            img_dst=data_root / "images" / split,
            lbl_dst=data_root / "labels" / split,
        )
        print(f"  Converted {n} images to YOLO format")

    # Write data.yaml
    yaml_path = data_root / "data.yaml"
    cfg = {
        "names": CLASSES,
        "nc": len(CLASSES),
        "train": "images/train",
        "val": "images/val",
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nDataset config written to {yaml_path}")
    print("Done!")


if __name__ == "__main__":
    main()
