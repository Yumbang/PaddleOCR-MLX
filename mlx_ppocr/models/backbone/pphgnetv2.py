"""PPHGNetV2 backbone (HGNetV2-L) for PP-OCRv5 server."""

import mlx.core as mx
import mlx.nn as nn

# HGNetV2-L architecture config
# (in_ch, mid_ch, out_ch, num_blocks, light, kernel, num_layers)
HGNETV2_L_STAGES = [
    (48, 48, 128, 1, False, 3, 6),
    (128, 96, 512, 1, False, 3, 6),
    (512, 192, 1024, 3, True, 5, 6),
    (1024, 384, 2048, 1, True, 5, 6),
]


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm + optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        groups: int = 1,
        bias: bool = False,
        act: str = "relu",
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias,
        )
        self.normalization = nn.BatchNorm(out_channels)
        self.act = act

    def __call__(self, x: mx.array) -> mx.array:
        x = self.convolution(x)
        x = self.normalization(x)
        if self.act == "relu":
            x = nn.relu(x)
        elif self.act == "silu":
            x = nn.silu(x)
        elif self.act == "none":
            pass
        return x


class LightConvBNAct(nn.Module):
    """Pointwise 1x1 conv (no activation) + depthwise kxk conv + BN + act."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, act: str = "relu"):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = ConvBNAct(in_channels, out_channels, 1, act="none")
        self.conv2 = ConvBNAct(
            out_channels, out_channels, kernel_size,
            padding=padding, groups=out_channels, act=act,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv2(self.conv1(x))


class HGBlock(nn.Module):
    """HGNetV2 core building block with feature aggregation."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_layers: int = 6,
        light: bool = False,
        act: str = "relu",
        identity: bool = False,
    ):
        super().__init__()
        self.identity = identity

        # Build sequential conv layers
        self.layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else mid_channels
            if light:
                self.layers.append(LightConvBNAct(in_ch, mid_channels, kernel_size, act=act))
            else:
                padding = kernel_size // 2
                self.layers.append(ConvBNAct(in_ch, mid_channels, kernel_size, padding=padding, act=act))

        # Aggregation: concat all outputs → squeeze → expand
        agg_channels = in_channels + num_layers * mid_channels
        self.aggregation = [
            ConvBNAct(agg_channels, out_channels // 2, 1, act=act),
            ConvBNAct(out_channels // 2, out_channels, 1, act=act),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        outs = [x]
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        # Concatenate along channel dim (NHWC)
        x = mx.concatenate(outs, axis=-1)
        x = self.aggregation[0](x)
        x = self.aggregation[1](x)
        return x


class HGStage(nn.Module):
    """One stage of HGNetV2 with optional downsampling + multiple HGBlocks."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_blocks: int,
        downsample: bool,
        downsample_stride: int | tuple = 2,
        light: bool = False,
        kernel_size: int = 3,
        num_layers: int = 6,
        act: str = "relu",
    ):
        super().__init__()
        self.has_downsample = downsample

        if downsample:
            ds = downsample_stride
            if isinstance(ds, (list, tuple)):
                padding = 1  # k=3, pad=k//2=1 for all dims
                stride = tuple(ds)
            else:
                padding = 1
                stride = ds
            self.downsample = ConvBNAct(
                in_channels, in_channels, 3,
                stride=stride, padding=padding,
                groups=in_channels, act="none",
            )

        self.blocks = []
        for i in range(num_blocks):
            block_in = in_channels if i == 0 else out_channels
            identity = i > 0
            self.blocks.append(
                HGBlock(block_in, mid_channels, out_channels, kernel_size, num_layers, light, act, identity)
            )

    def __call__(self, x: mx.array) -> mx.array:
        if self.has_downsample:
            x = self.downsample(x)
        for block in self.blocks:
            residual = x
            x = block(x)
            if block.identity:
                x = x + residual
        return x


class HGStem(nn.Module):
    """HGNetV2 stem block."""

    def __init__(self, stem_strides: list[int] | None = None):
        super().__init__()
        strides = stem_strides or [2, 1, 1, 2, 1]

        self.stem1 = ConvBNAct(3, 32, 3, stride=strides[0], padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.stem2a = ConvBNAct(32, 16, 2, stride=strides[1], padding=0)
        self.stem2b = ConvBNAct(16, 32, 2, stride=strides[2], padding=0)
        self.stem3 = ConvBNAct(64, 32, 3, stride=strides[3], padding=1)
        self.stem4 = ConvBNAct(32, 48, 1, stride=strides[4])

    def __call__(self, x: mx.array) -> mx.array:
        x = self.stem1(x)
        # Pad spatial dims by (0,1) on each side before stem2a and pool
        # Matches HF: F.pad(embedding, (0, 1, 0, 1)) in NCHW → pad H bottom, W right
        x_padded = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        x_2a = self.stem2a(x_padded)
        x_2a_padded = mx.pad(x_2a, [(0, 0), (0, 1), (0, 1), (0, 0)])
        x_2b = self.stem2b(x_2a_padded)
        x_pool = self.pool(x_padded)
        x = mx.concatenate([x_pool, x_2b], axis=-1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGEncoder(nn.Module):
    """Encoder wrapper holding stages (matches HF key prefix 'encoder.stages')."""

    def __init__(
        self,
        stage_downsample: list[bool],
        stage_downsample_strides: list,
        act: str,
    ):
        super().__init__()
        self.stages = []
        for i, (in_ch, mid_ch, out_ch, n_blocks, light, ks, n_layers) in enumerate(HGNETV2_L_STAGES):
            self.stages.append(
                HGStage(
                    in_ch, mid_ch, out_ch, n_blocks,
                    downsample=stage_downsample[i],
                    downsample_stride=stage_downsample_strides[i],
                    light=light,
                    kernel_size=ks,
                    num_layers=n_layers,
                    act=act,
                )
            )

    def __call__(self, x: mx.array) -> list[mx.array]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class PPHGNetV2(nn.Module):
    """PPHGNetV2 backbone for PP-OCRv5 server.

    Returns multi-scale features from all 4 stages.
    """

    def __init__(
        self,
        stage_downsample: list[bool] | None = None,
        stem_strides: list[int] | None = None,
        stage_downsample_strides: list | None = None,
        act: str = "relu",
    ):
        super().__init__()
        if stage_downsample is None:
            stage_downsample = [False, True, True, True]
        if stage_downsample_strides is None:
            stage_downsample_strides = [2, 2, 2, 2]

        self.embedder = HGStem(stem_strides)
        self.encoder = HGEncoder(stage_downsample, stage_downsample_strides, act)

    def __call__(self, x: mx.array) -> list[mx.array]:
        x = self.embedder(x)
        return self.encoder(x)
