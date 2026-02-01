"""Detection data augmentation transforms.

All transforms operate on ``dict(img=uint8_CHW, bboxes=Nx4_xyxy, cls=N)``.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Type alias for a sample dict
Sample = dict[str, Any]


# ---------------------------------------------------------------------------
# LetterBox
# ---------------------------------------------------------------------------


class LetterBox:
    """Resize longest side to *imgsz*, pad shorter side, adjust bboxes."""

    def __init__(self, imgsz: int = 640) -> None:
        if imgsz <= 0:
            raise ValueError(f"imgsz must be > 0, got {imgsz}")
        self.imgsz = imgsz

    def __call__(self, sample: Sample) -> Sample:
        img = sample["img"]  # (C, H, W) uint8
        bboxes = sample["bboxes"]  # (N, 4) xyxy
        cls = sample["cls"]

        _, h, w = img.shape
        scale = self.imgsz / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize via numpy (CHW → HWC for cv2)
        img_np = img.permute(1, 2, 0).numpy()
        img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to imgsz × imgsz
        pad_h = self.imgsz - new_h
        pad_w = self.imgsz - new_w
        top = pad_h // 2
        left = pad_w // 2

        img_padded = np.full((self.imgsz, self.imgsz, img.shape[0]), 114, dtype=np.uint8)
        img_padded[top : top + new_h, left : left + new_w] = img_resized

        img_out = torch.from_numpy(img_padded).permute(2, 0, 1)

        # Adjust bboxes
        if bboxes.numel() > 0:
            bboxes = bboxes.clone().float()
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + left
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + top
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clamp(0, self.imgsz)
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clamp(0, self.imgsz)

        return {"img": img_out, "bboxes": bboxes, "cls": cls}


# ---------------------------------------------------------------------------
# Mosaic
# ---------------------------------------------------------------------------

MOSAIC_FILL_VALUE = 114


class Mosaic:
    """2×2 mosaic augmentation: fetch 3 extra images, place in grid, merge labels."""

    def __init__(self, dataset: Any, imgsz: int = 640, p: float = 1.0) -> None:
        if imgsz <= 0:
            raise ValueError(f"imgsz must be > 0, got {imgsz}")
        if len(dataset) < 4:
            raise ValueError(f"Mosaic requires len(dataset) >= 4, got {len(dataset)}")
        self.dataset = dataset
        self.imgsz = imgsz
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample

        s = self.imgsz
        mosaic_img = np.full((s * 2, s * 2, 3), MOSAIC_FILL_VALUE, dtype=np.uint8)
        all_bboxes: list[torch.Tensor] = []
        all_cls: list[torch.Tensor] = []

        # Choose 3 random extra samples
        indices = [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        samples = [sample] + [self.dataset[i] for i in indices]

        # 2×2 grid placement offsets
        placements = [(0, 0), (s, 0), (0, s), (s, s)]

        for i, (samp, (ox, oy)) in enumerate(zip(samples, placements)):
            img_np = samp["img"].permute(1, 2, 0).numpy()  # (H, W, C)
            bboxes = samp["bboxes"].clone().float() if samp["bboxes"].numel() > 0 else samp["bboxes"]
            cls = samp["cls"].clone()

            h, w = img_np.shape[:2]
            # Scale to fit grid cell
            scale = s / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Place in mosaic
            mosaic_img[oy : oy + new_h, ox : ox + new_w] = img_resized

            # Adjust bboxes
            if bboxes.numel() > 0:
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + ox
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + oy
                all_bboxes.append(bboxes)
                all_cls.append(cls)

        # Resize mosaic from 2s×2s to s×s
        mosaic_img = cv2.resize(mosaic_img, (s, s), interpolation=cv2.INTER_LINEAR)
        img_out = torch.from_numpy(mosaic_img).permute(2, 0, 1)

        # Scale bboxes from 2s space to s space
        if all_bboxes:
            merged_bboxes = torch.cat(all_bboxes, 0)
            merged_bboxes *= 0.5  # 2s → s
            merged_bboxes[:, [0, 2]] = merged_bboxes[:, [0, 2]].clamp(0, s)
            merged_bboxes[:, [1, 3]] = merged_bboxes[:, [1, 3]].clamp(0, s)
            merged_cls = torch.cat(all_cls, 0)
        else:
            merged_bboxes = torch.zeros(0, 4)
            merged_cls = torch.zeros(0, dtype=torch.long)

        return {"img": img_out, "bboxes": merged_bboxes, "cls": merged_cls}


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------

MIXUP_BETA_ALPHA = 32.0
MIXUP_BETA_BETA = 32.0


class MixUp:
    """Beta-blend two images, concatenate labels."""

    def __init__(self, dataset: Any, p: float = 0.1) -> None:
        self.dataset = dataset
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample

        idx = random.randint(0, len(self.dataset) - 1)
        sample2 = self.dataset[idx]

        lam = np.random.beta(MIXUP_BETA_ALPHA, MIXUP_BETA_BETA)
        img1 = sample["img"].float()
        img2 = sample2["img"].float()

        # Resize img2 to match img1 if needed
        if img1.shape != img2.shape:
            _, h, w = img1.shape
            img2_np = sample2["img"].permute(1, 2, 0).numpy()
            img2_np = cv2.resize(img2_np, (w, h), interpolation=cv2.INTER_LINEAR)
            img2 = torch.from_numpy(img2_np).permute(2, 0, 1).float()

        mixed = (img1 * lam + img2 * (1 - lam)).clamp(0, 255).to(torch.uint8)

        # Concatenate labels
        bboxes_list = [sample["bboxes"]]
        cls_list = [sample["cls"]]
        if sample2["bboxes"].numel() > 0:
            bboxes_list.append(sample2["bboxes"])
            cls_list.append(sample2["cls"])

        if any(b.numel() > 0 for b in bboxes_list):
            merged_bboxes = torch.cat([b for b in bboxes_list if b.numel() > 0], 0)
            merged_cls = torch.cat([c for c in cls_list if c.numel() > 0], 0)
        else:
            merged_bboxes = torch.zeros(0, 4)
            merged_cls = torch.zeros(0, dtype=torch.long)

        return {"img": mixed, "bboxes": merged_bboxes, "cls": merged_cls}


# ---------------------------------------------------------------------------
# RandomPerspective
# ---------------------------------------------------------------------------

MIN_BBOX_AREA = 10.0
MAX_BBOX_ASPECT = 20.0


class RandomPerspective:
    """Affine transformation: rotate, translate, scale, shear."""

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def __call__(self, sample: Sample) -> Sample:
        img = sample["img"]  # (C, H, W)
        bboxes = sample["bboxes"]
        cls = sample["cls"]

        _, h, w = img.shape

        # Build affine matrix
        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        R = np.eye(3)
        a = self.rng.uniform(-self.degrees, self.degrees)
        s = self.rng.uniform(1 - self.scale, 1 + self.scale)
        cos_a, sin_a = math.cos(math.radians(a)), math.sin(math.radians(a))
        R[0, 0] = cos_a * s
        R[0, 1] = -sin_a * s
        R[1, 0] = sin_a * s
        R[1, 1] = cos_a * s

        S = np.eye(3)
        S[0, 1] = math.tan(math.radians(self.rng.uniform(-self.shear, self.shear)))
        S[1, 0] = math.tan(math.radians(self.rng.uniform(-self.shear, self.shear)))

        T = np.eye(3)
        T[0, 2] = self.rng.uniform(-self.translate, self.translate) * w + w / 2
        T[1, 2] = self.rng.uniform(-self.translate, self.translate) * h + h / 2

        M = T @ S @ R @ C
        M_2x3 = M[:2]

        # Warp image
        img_np = img.permute(1, 2, 0).numpy()
        warped = cv2.warpAffine(
            img_np, M_2x3, (w, h), borderValue=(114, 114, 114)
        )
        img_out = torch.from_numpy(warped).permute(2, 0, 1)

        # Transform bboxes
        if bboxes.numel() > 0:
            bboxes = bboxes.clone().float()
            n = bboxes.shape[0]
            # Get 4 corner points per box
            corners = torch.zeros(n, 4, 3)
            corners[:, 0, :2] = bboxes[:, :2]  # x1, y1
            corners[:, 1, 0] = bboxes[:, 2]
            corners[:, 1, 1] = bboxes[:, 1]  # x2, y1
            corners[:, 2, :2] = bboxes[:, 2:4]  # x2, y2
            corners[:, 3, 0] = bboxes[:, 0]
            corners[:, 3, 1] = bboxes[:, 3]  # x1, y2
            corners[:, :, 2] = 1.0

            M_t = torch.from_numpy(M_2x3).float()
            transformed = (corners.view(-1, 3) @ M_t.T).view(n, 4, 2)

            new_bboxes = torch.zeros(n, 4)
            new_bboxes[:, 0] = transformed[:, :, 0].min(1).values.clamp(0, w)
            new_bboxes[:, 1] = transformed[:, :, 1].min(1).values.clamp(0, h)
            new_bboxes[:, 2] = transformed[:, :, 0].max(1).values.clamp(0, w)
            new_bboxes[:, 3] = transformed[:, :, 1].max(1).values.clamp(0, h)

            # Filter by area and aspect ratio
            bw = new_bboxes[:, 2] - new_bboxes[:, 0]
            bh = new_bboxes[:, 3] - new_bboxes[:, 1]
            area = bw * bh
            aspect = torch.max(bw / (bh + 1e-6), bh / (bw + 1e-6))
            keep = (area > MIN_BBOX_AREA) & (aspect < MAX_BBOX_ASPECT)

            bboxes = new_bboxes[keep]
            cls = cls[keep] if cls.numel() > 0 else cls
        else:
            bboxes = torch.zeros(0, 4)

        return {"img": img_out, "bboxes": bboxes, "cls": cls}


# ---------------------------------------------------------------------------
# RandomHSV
# ---------------------------------------------------------------------------


class RandomHSV:
    """Jitter image in HSV color space."""

    def __init__(
        self,
        h: float = 0.015,
        s: float = 0.7,
        v: float = 0.4,
        seed: int | None = None,
    ) -> None:
        self.h = h
        self.s = s
        self.v = v
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def __call__(self, sample: Sample) -> Sample:
        img = sample["img"]  # (C, H, W) uint8

        if self.h == 0 and self.s == 0 and self.v == 0:
            return sample

        r = self.rng.uniform(-1, 1, 3) * [self.h, self.s, self.v] + 1
        img_np = img.permute(1, 2, 0).numpy()
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)

        hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)

        img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        sample = {**sample, "img": torch.from_numpy(img_np).permute(2, 0, 1)}
        return sample


# ---------------------------------------------------------------------------
# RandomFlip
# ---------------------------------------------------------------------------


class RandomFlip:
    """Horizontal flip with probability *p*."""

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        self.p = p
        self.rng = random.Random(seed) if seed is not None else random

    def __call__(self, sample: Sample) -> Sample:
        if self.rng.random() > self.p:
            return sample

        img = sample["img"]  # (C, H, W)
        bboxes = sample["bboxes"]

        img = img.flip(-1)  # horizontal flip

        if bboxes.numel() > 0:
            w = img.shape[-1]
            bboxes = bboxes.clone()
            x1 = bboxes[:, 0].clone()
            x2 = bboxes[:, 2].clone()
            bboxes[:, 0] = w - x2
            bboxes[:, 2] = w - x1

        return {"img": img.contiguous(), "bboxes": bboxes, "cls": sample["cls"]}


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


class Compose:
    """Sequential application of transforms."""

    def __init__(self, transforms: list[Any] | None = None) -> None:
        self.transforms: list[Any] = transforms if transforms is not None else []

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample
