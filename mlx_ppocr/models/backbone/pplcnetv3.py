"""PP-LCNetV3 backbone for PP-OCRv5 mobile recognition.

Ported from HuggingFace transformers PPLCNetV3 implementation.
Architecture: stem conv → 5 stages of depthwise-separable blocks
with learnable reparameterization and optional SE attention.
"""

import mlx.core as mx
import mlx.nn as nn

# PyTorch-compatible hard_sigmoid/hardswish (slope=1/6).
# Different from PaddlePaddle convention (slope=0.2) used in PPHGNetV2.
# Must match PyTorch since we load HF safetensors converted for PyTorch.

def _hard_sigmoid(x: mx.array) -> mx.array:
    return mx.clip(x / 6.0 + 0.5, 0.0, 1.0)


def _hardswish(x: mx.array) -> mx.array:
    return x * _hard_sigmoid(x)


def make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def _normalize_stride(stride) -> tuple[int, int]:
    """Normalize stride to (h, w) tuple."""
    if isinstance(stride, (list, tuple)):
        return tuple(stride)
    return (stride, stride)


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm + optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
        act: str | None = "hardswish",
    ):
        super().__init__()
        stride = _normalize_stride(stride)
        padding = kernel_size // 2
        self.convolution = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups, bias=False,
        )
        self.normalization = nn.BatchNorm(out_channels)
        self.act = act

    def __call__(self, x: mx.array) -> mx.array:
        x = self.convolution(x)
        x = self.normalization(x)
        if self.act == "hardswish":
            x = _hardswish(x)
        elif self.act == "silu":
            x = nn.silu(x)
        elif self.act == "relu":
            x = nn.relu(x)
        return x


class LearnableAffineBlock(nn.Module):
    """Learnable scale and bias: scale * x + bias."""

    def __init__(self):
        super().__init__()
        self.scale = mx.ones((1,))
        self.bias = mx.zeros((1,))

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x + self.bias


class ActLearnableAffineBlock(nn.Module):
    """Activation → learnable affine."""

    def __init__(self, act: str = "hardswish"):
        super().__init__()
        self._act = act
        self.lab = LearnableAffineBlock()

    def __call__(self, x: mx.array) -> mx.array:
        if self._act == "hardswish":
            x = _hardswish(x)
        elif self._act == "silu":
            x = nn.silu(x)
        return self.lab(x)


class LearnableRepLayer(nn.Module):
    """Multi-branch reparameterization layer.

    Sums: identity (BN) + conv_small_symmetric (1x1) + conv_symmetric[4] (kxk)
    Then: lab (affine) → act (only if stride != 2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int],
        groups: int = 1,
        num_conv_branches: int = 4,
        act: str = "hardswish",
    ):
        super().__init__()
        self.stride = _normalize_stride(stride)
        self.has_identity = (
            out_channels == in_channels and self.stride == (1, 1)
        )

        if self.has_identity:
            self.identity = nn.BatchNorm(in_channels)

        self.conv_symmetric = [
            ConvBNAct(in_channels, out_channels, kernel_size, self.stride,
                      groups=groups, act=None)
            for _ in range(num_conv_branches)
        ]

        self.conv_small_symmetric = (
            ConvBNAct(in_channels, out_channels, 1, self.stride,
                      groups=groups, act=None)
            if kernel_size > 1 else None
        )

        self.lab = LearnableAffineBlock()
        self.act = ActLearnableAffineBlock(act=act)

    def __call__(self, x: mx.array) -> mx.array:
        output = None

        if self.has_identity:
            output = self.identity(x)

        if self.conv_small_symmetric is not None:
            res = self.conv_small_symmetric(x)
            output = res if output is None else output + res

        for conv in self.conv_symmetric:
            res = conv(x)
            output = res if output is None else output + res

        x = self.lab(output)
        # Skip act only when stride is isotropic 2 (both dims)
        if self.stride != (2, 2):
            x = self.act(x)
        return x


class SqueezeExcitationModule(nn.Module):
    """SE module: global pool → fc → relu → fc → hard_sigmoid → multiply."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = channels // reduction
        # Using nn.Conv2d(1x1) with bias to match HF weight keys
        self.convolutions = [
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            None,  # relu placeholder
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            None,  # hard_sigmoid placeholder
        ]

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        # Global average pool: NHWC → N1C
        h = mx.mean(x, axis=(1, 2), keepdims=True)
        # Conv2d expects NHWC, h is [N, 1, 1, C] after keepdims
        h = self.convolutions[0](h)
        h = nn.relu(h)
        h = self.convolutions[2](h)
        h = _hard_sigmoid(h)
        return residual * h


class DepthwiseSeparableConvLayer(nn.Module):
    """Depthwise conv → SE (optional) → pointwise conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int],
        use_se: bool,
        act: str = "hardswish",
        num_conv_branches: int = 4,
        reduction: int = 4,
    ):
        super().__init__()
        self.depthwise_convolution = LearnableRepLayer(
            in_channels, in_channels, kernel_size, stride,
            groups=in_channels, num_conv_branches=num_conv_branches, act=act,
        )
        self.squeeze_excitation_module = (
            SqueezeExcitationModule(in_channels, reduction) if use_se else None
        )
        self.pointwise_convolution = LearnableRepLayer(
            in_channels, out_channels, 1, 1,
            groups=1, num_conv_branches=num_conv_branches, act=act,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.depthwise_convolution(x)
        if self.squeeze_excitation_module is not None:
            x = self.squeeze_excitation_module(x)
        x = self.pointwise_convolution(x)
        return x


class LCNetV3Block(nn.Module):
    """One stage of PP-LCNetV3: sequential depthwise-separable layers."""

    def __init__(self, block_configs: list, scale: float, divisor: int,
                 act: str, num_conv_branches: int, reduction: int):
        super().__init__()
        self.layers = []
        for kernel_size, in_ch, out_ch, stride, use_se in block_configs:
            scaled_in = make_divisible(in_ch * scale, divisor)
            scaled_out = make_divisible(out_ch * scale, divisor)
            self.layers.append(DepthwiseSeparableConvLayer(
                scaled_in, scaled_out, kernel_size, stride, use_se,
                act=act, num_conv_branches=num_conv_branches,
                reduction=reduction,
            ))

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


# Default block configs for PP-OCRv5 mobile rec
# (kernel_size, in_channels, out_channels, stride, use_se)
# Strides are rectangular for rec: [2,1] and [1,2] to handle text aspect ratio
LCNETV3_REC_BLOCK_CONFIGS = [
    [[3, 16, 32, 1, False]],
    [[3, 32, 64, 1, False], [3, 64, 64, 1, False]],
    [[3, 64, 128, [2, 1], False], [3, 128, 128, 1, False]],
    [
        [3, 128, 256, [1, 2], False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    [
        [5, 256, 512, [2, 1], True],
        [5, 512, 512, 1, True],
        [5, 512, 512, [2, 1], False],
        [5, 512, 512, 1, False],
    ],
]


class LCNetV3Encoder(nn.Module):
    """Encoder containing stem + blocks. Matches HF key prefix 'encoder.'."""

    def __init__(
        self,
        block_configs: list | None = None,
        scale: float = 0.95,
        divisor: int = 16,
        act: str = "hardswish",
        num_conv_branches: int = 4,
        reduction: int = 4,
    ):
        super().__init__()
        if block_configs is None:
            block_configs = LCNETV3_REC_BLOCK_CONFIGS

        stem_out = make_divisible(16 * scale, divisor)
        self.convolution = ConvBNAct(3, stem_out, kernel_size=3, stride=2, act=None)

        self.blocks = [
            LCNetV3Block(stage_cfg, scale, divisor, act, num_conv_branches,
                         reduction)
            for stage_cfg in block_configs
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.convolution(x)
        for block in self.blocks:
            x = block(x)
        return x


class PPLCNetV3(nn.Module):
    """PP-LCNetV3 backbone. Wraps encoder to match HF 'backbone.encoder.*' keys."""

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = LCNetV3Encoder(**kwargs)

    def __call__(self, x: mx.array) -> mx.array:
        return self.encoder(x)
