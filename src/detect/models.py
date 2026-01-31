"""YOLO detection model building blocks and constructors.

Primitives: Conv, DWConv, Bottleneck, C2f, C3/C3k, Attention, PSABlock,
C3k2, C2PSA, SPPF, DFL, Detect.

Model constructors: yolov8s, yolo11s, yolo26s (hardcoded, no YAML parsing).
Weight loading: load_ultralytics_weights for ultralytics checkpoint migration.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

GRID_CELL_OFFSET = 0.5


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """Compute same-padding for a given kernel size and dilation."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


# ---------------------------------------------------------------------------
# Core convolution blocks
# ---------------------------------------------------------------------------


class Conv(nn.Module):
    """Standard Conv2d + BatchNorm + SiLU activation."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act: nn.Module = self.default_act
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


# ---------------------------------------------------------------------------
# CSP bottleneck blocks
# ---------------------------------------------------------------------------


class Bottleneck(nn.Module):
    """Standard bottleneck: two convolutions with optional shortcut."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """YOLOv8 CSP block: cv1 splits, n bottlenecks, concat, cv2."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3 with configurable kernel size."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


# ---------------------------------------------------------------------------
# Attention blocks (YOLO11 / YOLO26)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention with depthwise positional encoding."""

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        qkv = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x + self.pe(v.reshape(B, -1, H, W)))
        return x


class PSABlock(nn.Module):
    """Attention + FFN with residual connections."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class C3k2(C2f):
    """C2f variant using C3k or Bottleneck branches (YOLO11/26)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C2PSA(nn.Module):
    """CSP with PSABlock chain (YOLO11/26)."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        if c1 != c2:
            raise ValueError(f"C2PSA requires c1 == c2, got c1={c1}, c2={c2}")
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        num_heads = max(1, self.c // 64)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=num_heads) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


# ---------------------------------------------------------------------------
# Spatial pooling
# ---------------------------------------------------------------------------


class SPPF(nn.Module):
    """Fast Spatial Pyramid Pooling (k=5 applied three times)."""

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


# ---------------------------------------------------------------------------
# Detection-specific modules
# ---------------------------------------------------------------------------


class DFL(nn.Module):
    """Distribution Focal Loss decode: fixed conv with arange weights."""

    def __init__(self, c1: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = x.view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def make_anchors(
    feats: list[torch.Tensor],
    strides: torch.Tensor,
    grid_cell_offset: float = GRID_CELL_OFFSET,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate anchor points and stride tensors from feature maps."""
    anchor_points: list[torch.Tensor] = []
    stride_tensor: list[torch.Tensor] = []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, dtype=dtype, device=device) + grid_cell_offset
        sy = torch.arange(end=h, dtype=dtype, device=device) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride.item(), dtype=dtype, device=device)
        )
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(
    distance: torch.Tensor,
    anchor_points: torch.Tensor,
    xywh: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """Transform distance (ltrb) to bounding box (xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(
    anchor_points: torch.Tensor,
    bbox: torch.Tensor,
    reg_max: float,
) -> torch.Tensor:
    """Transform bbox (xyxy) to distance (ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


# ---------------------------------------------------------------------------
# Detect head
# ---------------------------------------------------------------------------

MIN_DETECT_CHANNELS = 16


class Detect(nn.Module):
    """YOLO detection head: per-level box + cls branches with DFL decode."""

    def __init__(
        self,
        nc: int = 80,
        ch: tuple[int, ...] = (),
        reg_max: int = 16,
        end2end: bool = False,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.end2end = end2end
        self.stride = torch.zeros(self.nl)

        c2 = max(ch[0] if ch else MIN_DETECT_CHANNELS, self.reg_max * 4, MIN_DETECT_CHANNELS)
        c3 = max(ch[0] if ch else MIN_DETECT_CHANNELS, self.nc, MIN_DETECT_CHANNELS)

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        # Inference: decode predictions
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        anchors, strides = (
            t.transpose(0, 1) for t in make_anchors(x, self.stride, GRID_CELL_OFFSET)
        )
        # anchors: (2, A), strides: (1, A)

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), anchors.unsqueeze(0), xywh=False, dim=1)
        dbox = dbox * strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y.transpose(1, 2)  # (B, A, 4+nc)


# ---------------------------------------------------------------------------
# Full detection model
# ---------------------------------------------------------------------------

# Strides for P3/P4/P5 detection levels
DETECTION_STRIDES = (8.0, 16.0, 32.0)


class DetectionModel(nn.Module):
    """YOLO-family detection model with backbone, FPN+PAN neck, and Detect head.

    Submodule naming convention:
        b0..b9/b10 = backbone stages
        n1..n6     = neck stages
        detect     = detection head
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        # --- Backbone ---
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        p3 = self.b4(x)
        x = self.b5(p3)
        p4 = self.b6(x)
        x = self.b7(p4)
        x = self.b8(x)
        p5 = self.b9(x)
        if hasattr(self, "b10"):
            p5 = self.b10(p5)

        # --- Neck: FPN top-down ---
        up = F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        fpn_p4 = self.n1(torch.cat([up, p4], 1))

        up = F.interpolate(fpn_p4, size=p3.shape[2:], mode="nearest")
        fpn_p3 = self.n2(torch.cat([up, p3], 1))

        # --- Neck: PAN bottom-up ---
        down = self.n3(fpn_p3)
        pan_p4 = self.n4(torch.cat([down, fpn_p4], 1))

        down = self.n5(pan_p4)
        pan_p5 = self.n6(torch.cat([down, p5], 1))

        return self.detect([fpn_p3, pan_p4, pan_p5])


# ---------------------------------------------------------------------------
# Ultralytics index → our module name key maps
# ---------------------------------------------------------------------------

# YOLOv8: 23 layers (0-22). Upsample/Concat layers have no parameters.
_YOLOV8S_KEY_MAP: dict[int, str] = {
    0: "b0", 1: "b1", 2: "b2", 3: "b3", 4: "b4",
    5: "b5", 6: "b6", 7: "b7", 8: "b8", 9: "b9",
    12: "n1", 15: "n2", 16: "n3", 18: "n4", 19: "n5", 21: "n6",
    22: "detect",
}

# YOLO11: 24 layers (0-23). Adds C2PSA at index 10.
_YOLO11S_KEY_MAP: dict[int, str] = {
    0: "b0", 1: "b1", 2: "b2", 3: "b3", 4: "b4",
    5: "b5", 6: "b6", 7: "b7", 8: "b8", 9: "b9", 10: "b10",
    13: "n1", 16: "n2", 17: "n3", 19: "n4", 20: "n5", 22: "n6",
    23: "detect",
}

# YOLO26 uses the same layer layout as YOLO11.
_YOLO26S_KEY_MAP: dict[int, str] = _YOLO11S_KEY_MAP


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------


def _init_model(nc: int) -> DetectionModel:
    """Validate nc and create a bare DetectionModel instance."""
    if nc < 1:
        raise ValueError(f"nc must be >= 1, got {nc}")
    model = DetectionModel.__new__(DetectionModel)
    nn.Module.__init__(model)
    model.nc = nc
    return model


def _set_strides(model: DetectionModel) -> None:
    """Set detect head strides."""
    model.detect.stride = torch.tensor(DETECTION_STRIDES)


def yolov8s(nc: int = 80) -> DetectionModel:
    """YOLOv8-S: depth=0.33, width=0.50, channels [32,64,128,256,512], C2f blocks."""
    model = _init_model(nc)

    # Backbone
    model.b0 = Conv(3, 32, 3, 2)
    model.b1 = Conv(32, 64, 3, 2)
    model.b2 = C2f(64, 64, n=1)
    model.b3 = Conv(64, 128, 3, 2)
    model.b4 = C2f(128, 128, n=2)
    model.b5 = Conv(128, 256, 3, 2)
    model.b6 = C2f(256, 256, n=2)
    model.b7 = Conv(256, 512, 3, 2)
    model.b8 = C2f(512, 512, n=1)
    model.b9 = SPPF(512, 512, 5)

    # Neck — FPN
    model.n1 = C2f(512 + 256, 256, n=1)
    model.n2 = C2f(256 + 128, 128, n=1)
    # Neck — PAN
    model.n3 = Conv(128, 128, 3, 2)
    model.n4 = C2f(128 + 256, 256, n=1)
    model.n5 = Conv(256, 256, 3, 2)
    model.n6 = C2f(256 + 512, 512, n=1)

    # Detect
    model.detect = Detect(nc, ch=(128, 256, 512))
    model._ultralytics_key_map = _YOLOV8S_KEY_MAP
    _set_strides(model)
    return model


def yolo11s(nc: int = 80) -> DetectionModel:
    """YOLO11-S: depth=0.50, width=0.50, C3k2 + C2PSA blocks."""
    model = _init_model(nc)

    # Backbone
    model.b0 = Conv(3, 32, 3, 2)
    model.b1 = Conv(32, 64, 3, 2)
    model.b2 = C3k2(64, 64, n=1, c3k=False)
    model.b3 = Conv(64, 128, 3, 2)
    model.b4 = C3k2(128, 128, n=1, c3k=False)
    model.b5 = Conv(128, 256, 3, 2)
    model.b6 = C3k2(256, 256, n=1, c3k=True)
    model.b7 = Conv(256, 512, 3, 2)
    model.b8 = C3k2(512, 512, n=1, c3k=True)
    model.b9 = SPPF(512, 512, 5)
    model.b10 = C2PSA(512, 512, n=1)

    # Neck — FPN
    model.n1 = C3k2(512 + 256, 256, n=1, c3k=False)
    model.n2 = C3k2(256 + 128, 128, n=1, c3k=False)
    # Neck — PAN
    model.n3 = Conv(128, 128, 3, 2)
    model.n4 = C3k2(128 + 256, 256, n=1, c3k=False)
    model.n5 = Conv(256, 256, 3, 2)
    model.n6 = C3k2(256 + 512, 512, n=1, c3k=True)

    # Detect
    model.detect = Detect(nc, ch=(128, 256, 512))
    model._ultralytics_key_map = _YOLO11S_KEY_MAP
    _set_strides(model)
    return model


def yolo26s(nc: int = 80) -> DetectionModel:
    """YOLO26-S: depth=0.50, width=0.50, reg_max=1, end2end=True."""
    model = _init_model(nc)

    # Backbone (same as yolo11s)
    model.b0 = Conv(3, 32, 3, 2)
    model.b1 = Conv(32, 64, 3, 2)
    model.b2 = C3k2(64, 64, n=1, c3k=False)
    model.b3 = Conv(64, 128, 3, 2)
    model.b4 = C3k2(128, 128, n=1, c3k=False)
    model.b5 = Conv(128, 256, 3, 2)
    model.b6 = C3k2(256, 256, n=1, c3k=True)
    model.b7 = Conv(256, 512, 3, 2)
    model.b8 = C3k2(512, 512, n=1, c3k=True)
    model.b9 = SPPF(512, 512, 5)
    model.b10 = C2PSA(512, 512, n=1)

    # Neck — FPN
    model.n1 = C3k2(512 + 256, 256, n=1, c3k=False)
    model.n2 = C3k2(256 + 128, 128, n=1, c3k=False)
    # Neck — PAN
    model.n3 = Conv(128, 128, 3, 2)
    model.n4 = C3k2(128 + 256, 256, n=1, c3k=False)
    model.n5 = Conv(256, 256, 3, 2)
    model.n6 = C3k2(256 + 512, 512, n=1, c3k=True)

    # Detect with reg_max=1 (direct regression) and end2end (NMS-free)
    model.detect = Detect(nc, ch=(128, 256, 512), reg_max=1, end2end=True)
    model._ultralytics_key_map = _YOLO26S_KEY_MAP
    _set_strides(model)
    return model


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def load_ultralytics_weights(
    model: DetectionModel, pt_path: str | Path
) -> list[str]:
    """Load weights from an ultralytics-format state dict file.

    The checkpoint must be a pure state dict (``torch.save(model.state_dict(), path)``).
    To convert a legacy ultralytics ``.pt``::

        import torch
        ckpt = torch.load("model.pt", map_location="cpu")
        sd = ckpt["model"].state_dict()
        torch.save(sd, "model_sd.pt")

    Key mapping (ultralytics ``model.model.{idx}.rest`` → our ``{module}.rest``)::

        YOLOv8s: 0-9 → b0-b9, 12→n1, 15→n2, 16→n3, 18→n4, 19→n5, 21→n6, 22→detect
        YOLO11s: 0-10 → b0-b10, 13→n1, 16→n2, 17→n3, 19→n4, 20→n5, 22→n6, 23→detect
        YOLO26s: same layout as YOLO11s

    Returns:
        List of unmapped key names from the checkpoint (if any).
    """
    pt_path = Path(pt_path)
    if not pt_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {pt_path}")

    state_dict: dict[str, Any] = torch.load(pt_path, map_location="cpu", weights_only=True)

    key_map: dict[int, str] = getattr(model, "_ultralytics_key_map", {})

    mapped_sd: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for key, value in state_dict.items():
        # Expected format: "model.model.{idx}.{rest}" or just "{idx}.{rest}"
        parts = key.split(".")
        try:
            if parts[0] == "model" and parts[1] == "model":
                idx = int(parts[2])
                rest = ".".join(parts[3:])
            else:
                idx = int(parts[0])
                rest = ".".join(parts[1:])
        except (ValueError, IndexError):
            unmapped.append(key)
            continue

        if idx in key_map:
            new_key = f"{key_map[idx]}.{rest}"
            mapped_sd[new_key] = value
        else:
            unmapped.append(key)

    if unmapped:
        logger.warning("Unmapped keys during weight loading: %s", unmapped[:20])

    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    if missing:
        logger.warning("Missing keys in model: %s", missing[:20])
    if unexpected:
        logger.warning("Unexpected keys from checkpoint: %s", unexpected[:20])

    return unmapped
