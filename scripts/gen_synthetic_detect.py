"""Generate a small synthetic YOLO-format detection dataset for smoke-testing."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import yaml


def main(
    root: str | Path = "data/synthetic_detect",
    num_train: int = 64,
    num_val: int = 16,
    imgsz: int = 128,
    nc: int = 5,
    max_boxes: int = 6,
    seed: int = 42,
) -> Path:
    root = Path(root)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    for split, n_images in [("train", num_train), ("val", num_val)]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_images):
            # Random background
            img = np_rng.randint(80, 180, (imgsz, imgsz, 3), dtype=np.uint8)
            labels: list[str] = []
            n_boxes = rng.randint(1, max_boxes)

            for _ in range(n_boxes):
                cls_id = rng.randint(0, nc - 1)
                # Random box (ensure minimum size)
                cx = rng.uniform(0.15, 0.85)
                cy = rng.uniform(0.15, 0.85)
                bw = rng.uniform(0.08, 0.4)
                bh = rng.uniform(0.08, 0.4)
                # Clamp to image bounds
                bw = min(bw, cx * 2, (1 - cx) * 2)
                bh = min(bh, cy * 2, (1 - cy) * 2)

                # Draw rectangle on image
                x1_px = int((cx - bw / 2) * imgsz)
                y1_px = int((cy - bh / 2) * imgsz)
                x2_px = int((cx + bw / 2) * imgsz)
                y2_px = int((cy + bh / 2) * imgsz)
                color = (
                    int(np_rng.randint(0, 255)),
                    int(np_rng.randint(0, 255)),
                    int(np_rng.randint(0, 255)),
                )
                cv2.rectangle(img, (x1_px, y1_px), (x2_px, y2_px), color, -1)
                labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Save image (BGR for cv2)
            cv2.imwrite(str(img_dir / f"{i:04d}.jpg"), img)
            # Save label
            (lbl_dir / f"{i:04d}.txt").write_text("\n".join(labels) + "\n")

    # Write data YAML
    data_yaml = root / "data.yaml"
    cfg = {
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": [f"class_{i}" for i in range(nc)],
    }
    with open(data_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"Synthetic dataset created at {root}")
    print(f"  train: {num_train} images, val: {num_val} images, nc: {nc}")
    print(f"  YAML: {data_yaml}")
    return root


if __name__ == "__main__":
    main()
