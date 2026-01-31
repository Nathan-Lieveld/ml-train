"""Comprehensive tests for src/detect/models.py detection building blocks and constructors."""

from __future__ import annotations

import logging

import pytest
import torch

from src.detect.models import (
    Attention,
    Bottleneck,
    C2f,
    C2PSA,
    C3,
    C3k,
    C3k2,
    Conv,
    DFL,
    Detect,
    DWConv,
    PSABlock,
    SPPF,
    autopad,
    bbox2dist,
    dist2bbox,
    load_ultralytics_weights,
    make_anchors,
    yolo11s,
    yolo26s,
    yolov8s,
)

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class TestConv:
    """Tests for Conv (Conv2d + BN + SiLU)."""

    def test_output_shape(self):
        conv = Conv(3, 16, k=3, s=1)
        x = torch.randn(1, 3, 32, 32)
        out = conv(x)
        assert out.shape == (1, 16, 32, 32)

    def test_output_shape_stride2(self):
        conv = Conv(16, 32, k=3, s=2)
        x = torch.randn(2, 16, 64, 64)
        out = conv(x)
        assert out.shape == (2, 32, 32, 32)

    @pytest.mark.parametrize("k", [1, 3, 5, 7])
    def test_autopad_same_padding(self, k: int):
        p = autopad(k)
        assert p == k // 2

    @pytest.mark.parametrize("k,d,expected", [(3, 1, 1), (3, 2, 2), (5, 1, 2)])
    def test_autopad_dilation(self, k: int, d: int, expected: int):
        p = autopad(k, d=d)
        assert p == expected

    def test_bn_and_silu_applied(self):
        conv = Conv(3, 8, k=1, s=1, act=True)
        # BN is present
        assert isinstance(conv.bn, torch.nn.BatchNorm2d)
        # SiLU activation is present
        assert isinstance(conv.act, torch.nn.SiLU)

    def test_act_false_gives_identity(self):
        conv = Conv(3, 8, k=1, s=1, act=False)
        assert isinstance(conv.act, torch.nn.Identity)

    def test_custom_activation(self):
        conv = Conv(3, 8, k=1, s=1, act=torch.nn.ReLU())
        assert isinstance(conv.act, torch.nn.ReLU)


class TestDWConv:
    """Tests for DWConv (depthwise convolution)."""

    def test_output_channels(self):
        dw = DWConv(16, 16, k=3, s=1)
        x = torch.randn(1, 16, 32, 32)
        out = dw(x)
        assert out.shape[1] == 16

    def test_depthwise_groups(self):
        dw = DWConv(16, 16, k=3, s=1)
        # groups should equal gcd(c1, c2) = 16 for depthwise
        assert dw.conv.groups == 16

    def test_different_channels_gcd(self):
        dw = DWConv(12, 8, k=3, s=1)
        import math

        assert dw.conv.groups == math.gcd(12, 8)


class TestBottleneck:
    """Tests for Bottleneck block."""

    def test_shortcut_add_when_channels_match(self):
        bn = Bottleneck(64, 64, shortcut=True)
        assert bn.add is True
        x = torch.randn(1, 64, 16, 16)
        out = bn(x)
        assert out.shape == x.shape

    def test_no_shortcut_when_false(self):
        bn = Bottleneck(64, 64, shortcut=False)
        assert bn.add is False

    def test_no_shortcut_when_channels_differ(self):
        bn = Bottleneck(32, 64, shortcut=True)
        # shortcut is True but c1 != c2, so add should be False
        assert bn.add is False

    def test_output_shape(self):
        bn = Bottleneck(64, 128, shortcut=False, e=0.5)
        x = torch.randn(2, 64, 8, 8)
        out = bn(x)
        assert out.shape == (2, 128, 8, 8)


class TestC2f:
    """Tests for C2f (YOLOv8 CSP block)."""

    def test_output_channels(self):
        block = C2f(64, 128, n=1)
        x = torch.randn(1, 64, 16, 16)
        out = block(x)
        assert out.shape == (1, 128, 16, 16)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_varying_n(self, n: int):
        block = C2f(64, 64, n=n)
        x = torch.randn(1, 64, 16, 16)
        out = block(x)
        assert out.shape == (1, 64, 16, 16)

    def test_split_concat_preserves_spatial(self):
        block = C2f(32, 32, n=2, shortcut=True)
        x = torch.randn(1, 32, 20, 20)
        out = block(x)
        assert out.shape[2:] == x.shape[2:]


class TestC3k:
    """Tests for C3k (C3 with configurable kernel)."""

    @pytest.mark.parametrize("k", [3, 5])
    def test_configurable_kernel(self, k: int):
        block = C3k(64, 64, n=1, k=k)
        x = torch.randn(1, 64, 16, 16)
        out = block(x)
        assert out.shape == (1, 64, 16, 16)

    def test_inherits_c3(self):
        assert issubclass(C3k, C3)


class TestAttention:
    """Tests for multi-head self-attention."""

    def test_output_shape(self):
        attn = Attention(dim=64, num_heads=4, attn_ratio=0.5)
        x = torch.randn(1, 64, 8, 8)
        out = attn(x)
        assert out.shape == x.shape

    def test_attention_softmax(self):
        """Verify attention weights sum to 1 along the key dimension."""
        attn = Attention(dim=64, num_heads=4, attn_ratio=0.5)
        x = torch.randn(1, 64, 4, 4)
        B, C, H, W = x.shape
        N = H * W

        qkv = attn.qkv(x)
        qkv = qkv.view(B, attn.num_heads, attn.key_dim * 2 + attn.head_dim, N)
        q, k, _v = qkv.split([attn.key_dim, attn.key_dim, attn.head_dim], dim=2)

        weights = (q.transpose(-2, -1) @ k) * attn.scale
        weights = weights.softmax(dim=-1)
        # Each row of attention should sum to 1
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_batch_size(self):
        attn = Attention(dim=128, num_heads=8, attn_ratio=0.5)
        x = torch.randn(4, 128, 8, 8)
        out = attn(x)
        assert out.shape == (4, 128, 8, 8)


class TestPSABlock:
    """Tests for PSABlock (Attention + FFN with residual)."""

    def test_residual_output_shape(self):
        psa = PSABlock(c=64, attn_ratio=0.5, num_heads=4)
        x = torch.randn(1, 64, 8, 8)
        out = psa(x)
        assert out.shape == x.shape

    def test_ffn_hidden_dim(self):
        psa = PSABlock(c=64, attn_ratio=0.5, num_heads=4)
        # FFN is Sequential(Conv(c, c*2, 1), Conv(c*2, c, 1, act=False))
        ffn_first_conv = psa.ffn[0]
        ffn_second_conv = psa.ffn[1]
        # Hidden dim = 2 * embed_dim
        assert ffn_first_conv.conv.out_channels == 128  # 2 * 64
        assert ffn_second_conv.conv.out_channels == 64


class TestC3k2:
    """Tests for C3k2 (C2f variant with C3k or Bottleneck branches)."""

    def test_forward_bottleneck_branch(self):
        block = C3k2(64, 64, n=1, c3k=False)
        x = torch.randn(1, 64, 16, 16)
        out = block(x)
        assert out.shape == (1, 64, 16, 16)
        # When c3k=False, inner modules are Bottleneck
        assert isinstance(block.m[0], Bottleneck)

    def test_forward_c3k_branch(self):
        block = C3k2(64, 64, n=1, c3k=True)
        x = torch.randn(1, 64, 16, 16)
        out = block(x)
        assert out.shape == (1, 64, 16, 16)
        # When c3k=True, inner modules are C3k
        assert isinstance(block.m[0], C3k)

    def test_inherits_c2f(self):
        assert issubclass(C3k2, C2f)


class TestC2PSA:
    """Tests for C2PSA (CSP with PSABlock chain)."""

    def test_output_shape(self):
        block = C2PSA(128, 128, n=1)
        x = torch.randn(1, 128, 8, 8)
        out = block(x)
        assert out.shape == (1, 128, 8, 8)

    def test_requires_c1_eq_c2(self):
        with pytest.raises(ValueError, match="C2PSA requires c1 == c2"):
            C2PSA(64, 128)

    def test_chain_of_psablocks(self):
        block = C2PSA(128, 128, n=3)
        assert len(block.m) == 3
        for sub in block.m:
            assert isinstance(sub, PSABlock)


class TestSPPF:
    """Tests for SPPF (Spatial Pyramid Pooling Fast)."""

    def test_output_channels(self):
        sppf = SPPF(64, 128, k=5)
        x = torch.randn(1, 64, 16, 16)
        out = sppf(x)
        assert out.shape[1] == 128

    def test_spatial_dims_preserved(self):
        sppf = SPPF(64, 128, k=5)
        x = torch.randn(1, 64, 20, 20)
        out = sppf(x)
        assert out.shape[2:] == (20, 20)


class TestDFL:
    """Tests for DFL (Distribution Focal Loss decode)."""

    def test_output_dim_is_4(self):
        dfl = DFL(c1=16)
        # Input shape: (B, 4*reg_max, A) where A is num anchors
        x = torch.randn(1, 4 * 16, 100)
        out = dfl(x)
        assert out.shape == (1, 4, 100)

    def test_weights_are_fixed_arange(self):
        dfl = DFL(c1=16)
        expected = torch.arange(16, dtype=torch.float).view(1, 16, 1, 1)
        assert torch.equal(dfl.conv.weight.data, expected)

    def test_weights_not_trainable(self):
        dfl = DFL(c1=16)
        assert not dfl.conv.weight.requires_grad


class TestDetect:
    """Tests for Detect head."""

    def test_per_level_output_shapes_train(self):
        detect = Detect(nc=80, ch=(128, 256, 512), reg_max=16)
        detect.stride = torch.tensor([8.0, 16.0, 32.0])
        detect.train()
        feats = [
            torch.randn(1, 128, 80, 80),
            torch.randn(1, 256, 40, 40),
            torch.randn(1, 512, 20, 20),
        ]
        out = detect(feats)
        # In training mode returns a list
        assert isinstance(out, list)
        assert len(out) == 3
        no = 80 + 16 * 4
        assert out[0].shape == (1, no, 80, 80)
        assert out[1].shape == (1, no, 40, 40)
        assert out[2].shape == (1, no, 20, 20)

    def test_decode_output_shape_eval(self):
        nc = 80
        reg_max = 16
        detect = Detect(nc=nc, ch=(128, 256, 512), reg_max=reg_max)
        detect.stride = torch.tensor([8.0, 16.0, 32.0])
        detect.eval()
        feats = [
            torch.randn(1, 128, 80, 80),
            torch.randn(1, 256, 40, 40),
            torch.randn(1, 512, 20, 20),
        ]
        out = detect(feats)
        total_anchors = 80 * 80 + 40 * 40 + 20 * 20  # 8400
        assert out.shape == (1, total_anchors, 4 + nc)

    def test_reg_max_1_identity_dfl(self):
        detect = Detect(nc=10, ch=(32,), reg_max=1)
        assert isinstance(detect.dfl, torch.nn.Identity)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestMakeAnchors:
    def test_shapes(self):
        feats = [
            torch.randn(1, 128, 80, 80),
            torch.randn(1, 256, 40, 40),
            torch.randn(1, 512, 20, 20),
        ]
        strides = torch.tensor([8.0, 16.0, 32.0])
        anchors, stride_t = make_anchors(feats, strides)
        total = 80 * 80 + 40 * 40 + 20 * 20
        assert anchors.shape == (total, 2)
        assert stride_t.shape == (total, 1)


class TestDist2Bbox:
    def test_xywh_output(self):
        distance = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]).permute(0, 2, 1)
        anchor = torch.tensor([[5.0, 5.0]])
        out = dist2bbox(distance, anchor.unsqueeze(0).permute(0, 2, 1), xywh=True, dim=1)
        assert out.shape[1] == 4

    def test_xyxy_output(self):
        distance = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]).permute(0, 2, 1)
        anchor = torch.tensor([[5.0, 5.0]])
        out = dist2bbox(
            distance, anchor.unsqueeze(0).permute(0, 2, 1), xywh=False, dim=1
        )
        assert out.shape[1] == 4


class TestBbox2Dist:
    def test_clamp(self):
        anchor = torch.tensor([[5.0, 5.0]])
        bbox = torch.tensor([[2.0, 2.0, 8.0, 8.0]])
        out = bbox2dist(anchor, bbox, reg_max=16)
        assert out.shape == (1, 4)
        assert (out >= 0).all()


# ---------------------------------------------------------------------------
# Model constructors (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestYolov8sForward:
    def test_forward_shape(self):
        model = yolov8s(nc=80)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 8400, 84)


@pytest.mark.slow
class TestYolo11sForward:
    def test_forward_shape(self):
        model = yolo11s(nc=80)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 8400, 84)


@pytest.mark.slow
class TestYolo26sForward:
    def test_forward_shape(self):
        """yolo26s uses reg_max=1, output is still (B, A, 4+nc)."""
        model = yolo26s(nc=80)
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 8400, 84)


@pytest.mark.parametrize("nc", [1, 80, 1203])
def test_constructor_nc(nc: int):
    """Various nc values produce valid models."""
    model = yolov8s(nc=nc)
    assert model.nc == nc
    assert model.detect.nc == nc


@pytest.mark.parametrize(
    "factory", [yolov8s, yolo11s, yolo26s], ids=["yolov8s", "yolo11s", "yolo26s"]
)
def test_param_count_reasonable(factory):
    """Each model should have < 15M parameters."""
    model = factory(nc=80)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 15_000_000, (
        f"{factory.__name__} has {param_count:,} params (limit 15M)"
    )


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


class TestWeightLoading:
    def test_load_weights_warns_on_unmapped(self, tmp_path, caplog):
        """Fabricated state dict with extra keys logs a warning."""
        model = yolov8s(nc=80)
        sd: dict[str, torch.Tensor] = {}
        # Add valid mapped keys from the model (index 0 maps to b0)
        for name, param in model.b0.named_parameters():
            sd[f"0.{name}"] = param.data.clone()
        # Add unmapped keys (index 99 is not in the key map)
        sd["99.conv.weight"] = torch.randn(16, 3, 3, 3)
        sd["99.bn.weight"] = torch.randn(16)

        pt_path = tmp_path / "fake_weights.pt"
        torch.save(sd, pt_path)

        with caplog.at_level(logging.WARNING, logger="src.detect.models"):
            unmapped = load_ultralytics_weights(model, pt_path)

        assert len(unmapped) > 0
        assert any("99.conv.weight" in k for k in unmapped)

    def test_load_weights_missing_keys(self, tmp_path):
        """Partial state dict loads successfully with strict=False."""
        model = yolov8s(nc=80)
        sd: dict[str, torch.Tensor] = {}
        # Only provide weights for backbone stage 0 (index 0 -> b0)
        for name, param in model.b0.named_parameters():
            sd[f"0.{name}"] = param.data.clone()

        pt_path = tmp_path / "partial_weights.pt"
        torch.save(sd, pt_path)

        # Should not raise even though most keys are missing
        unmapped = load_ultralytics_weights(model, pt_path)
        assert isinstance(unmapped, list)

    def test_load_weights_file_not_found(self, tmp_path):
        model = yolov8s(nc=80)
        with pytest.raises(FileNotFoundError):
            load_ultralytics_weights(model, tmp_path / "nonexistent.pt")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nc", [0, -1])
def test_invalid_nc_raises(nc: int):
    """nc=0 and nc=-1 should raise ValueError."""
    with pytest.raises(ValueError, match="nc must be >= 1"):
        yolov8s(nc=nc)

    with pytest.raises(ValueError, match="nc must be >= 1"):
        yolo11s(nc=nc)

    with pytest.raises(ValueError, match="nc must be >= 1"):
        yolo26s(nc=nc)


def test_single_pixel_input():
    """(1, 3, 1, 1) should not crash Conv or Bottleneck primitives."""
    conv = Conv(3, 16, k=1, s=1)
    conv.eval()  # BN requires >1 spatial elements in train mode
    x = torch.randn(1, 3, 1, 1)
    with torch.no_grad():
        out = conv(x)
    assert out.shape == (1, 16, 1, 1)


def test_non_square_input():
    """Non-square input (1, 3, 480, 640) through primitives."""
    conv = Conv(3, 16, k=3, s=1)
    x = torch.randn(1, 3, 480, 640)
    out = conv(x)
    assert out.shape == (1, 16, 480, 640)

    c2f = C2f(16, 32, n=1)
    out2 = c2f(out)
    assert out2.shape == (1, 32, 480, 640)


@pytest.mark.slow
def test_non_square_full_model():
    """Non-square input through full detection model."""
    model = yolov8s(nc=80)
    model.eval()
    x = torch.randn(1, 3, 480, 640)
    with torch.no_grad():
        out = model(x)
    # P3: 60x80=4800, P4: 30x40=1200, P5: 15x20=300 -> total 6300
    assert out.shape == (1, 6300, 84)
