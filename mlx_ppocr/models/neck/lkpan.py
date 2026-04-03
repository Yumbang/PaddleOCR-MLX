"""LKPAN neck (Large Kernel PAN) with IntraClass blocks for PP-OCRv5 server detection."""

import mlx.core as mx
import mlx.nn as nn

from mlx_ppocr.utils.ops import nearest_upsample_to


class IntraClassBlock(nn.Module):
    """Directional convolution block for intra-class feature learning."""

    def __init__(self, in_channels: int = 64, reduce_channels: int = 32):
        super().__init__()
        self.conv_reduce_channel = nn.Conv2d(in_channels, reduce_channels, 1, bias=True)

        self.vertical_long_to_small_conv_longratio = nn.Conv2d(
            reduce_channels, reduce_channels, (7, 1), padding=(3, 0), bias=True,
        )
        self.vertical_long_to_small_conv_midratio = nn.Conv2d(
            reduce_channels, reduce_channels, (5, 1), padding=(2, 0), bias=True,
        )
        self.vertical_long_to_small_conv_shortratio = nn.Conv2d(
            reduce_channels, reduce_channels, (3, 1), padding=(1, 0), bias=True,
        )

        self.horizontal_small_to_long_conv_longratio = nn.Conv2d(
            reduce_channels, reduce_channels, (1, 7), padding=(0, 3), bias=True,
        )
        self.horizontal_small_to_long_conv_midratio = nn.Conv2d(
            reduce_channels, reduce_channels, (1, 5), padding=(0, 2), bias=True,
        )
        self.horizontal_small_to_long_conv_shortratio = nn.Conv2d(
            reduce_channels, reduce_channels, (1, 3), padding=(0, 1), bias=True,
        )

        self.symmetric_conv_long_longratio = nn.Conv2d(
            reduce_channels, reduce_channels, (7, 7), padding=(3, 3), bias=True,
        )
        self.symmetric_conv_long_midratio = nn.Conv2d(
            reduce_channels, reduce_channels, (5, 5), padding=(2, 2), bias=True,
        )
        self.symmetric_conv_long_shortratio = nn.Conv2d(
            reduce_channels, reduce_channels, (3, 3), padding=(1, 1), bias=True,
        )

        self.conv_final = ConvBNAct(reduce_channels, in_channels, 1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv_reduce_channel(x)

        v_long = self.vertical_long_to_small_conv_longratio(h)
        v_mid = self.vertical_long_to_small_conv_midratio(v_long)
        v_short = self.vertical_long_to_small_conv_shortratio(v_mid)

        h_long = self.horizontal_small_to_long_conv_longratio(h)
        h_mid = self.horizontal_small_to_long_conv_midratio(h_long)
        h_short = self.horizontal_small_to_long_conv_shortratio(h_mid)

        s_long = self.symmetric_conv_long_longratio(h)
        s_mid = self.symmetric_conv_long_midratio(s_long)
        s_short = self.symmetric_conv_long_shortratio(s_mid)

        h = v_short + h_short + s_short
        h = self.conv_final(h)
        return x + h


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int, bias: bool = False):
        super().__init__()
        self.convolution = nn.Conv2d(in_ch, out_ch, ks, bias=bias)
        self.norm = nn.BatchNorm(out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.norm(self.convolution(x)))


class LKPAN(nn.Module):
    """Large Kernel PAN neck for PP-OCRv5 server detection.

    Flow (matching HF Transformers):
    1. Channel adjustment: 4 × 1x1 conv → 256 ch
    2. Top-down FPN: upsample 2x + add (high→low resolution)
    3. Feature projection: 4 × 9x9 conv → 64 ch
    4. Bottom-up PAN: stride-2 3x3 conv + add (low→high resolution)
    5. Lateral refinement: 4 × 9x9 conv
    6. Intra-class blocks
    7. Upsample all to largest, concat reversed
    """

    def __init__(
        self,
        in_channels: list[int] = [128, 512, 1024, 2048],
        neck_out_channels: int = 256,
    ):
        super().__init__()
        inner_channels = neck_out_channels // 4  # 64

        # Step 1: 1x1 conv to unify channels → 256
        self.input_channel_adjustment_convolution = [
            nn.Conv2d(ch, neck_out_channels, 1, bias=False) for ch in in_channels
        ]

        # Step 3: 9x9 feature projection → 64
        self.input_feature_projection_convolution = [
            nn.Conv2d(neck_out_channels, inner_channels, 9, padding=4, bias=False)
            for _ in range(4)
        ]

        # Step 4: stride-2 3x3 conv for bottom-up downsampling
        self.path_aggregation_head_convolution = [
            nn.Conv2d(inner_channels, inner_channels, 3, stride=2, padding=1, bias=False)
            for _ in range(3)
        ]

        # Step 5: 9x9 lateral refinement
        self.path_aggregation_lateral_convolution = [
            nn.Conv2d(inner_channels, inner_channels, 9, padding=4, bias=False)
            for _ in range(4)
        ]

        # Step 6: Intra-class blocks
        self.intraclass_blocks = [
            IntraClassBlock(inner_channels, inner_channels // 2)
            for _ in range(4)
        ]

        self.scale_factor_list = [1, 2, 4, 8]

    def __call__(self, features: list[mx.array]) -> mx.array:
        # Step 1: Channel adjustment
        adjusted = [
            self.input_channel_adjustment_convolution[i](features[i])
            for i in range(4)
        ]

        # Step 2: Top-down FPN (from smallest to largest)
        top_down = [None] * 4
        top_down[3] = adjusted[3]
        for i in range(2, -1, -1):
            target_h, target_w = adjusted[i].shape[1], adjusted[i].shape[2]
            up = nearest_upsample_to(top_down[i + 1], target_h, target_w)
            top_down[i] = adjusted[i] + up

        # Step 3: Feature projection (9x9 conv)
        projected = []
        for i in range(4):
            # Last level uses adjusted, others use top_down
            inp = top_down[i] if i < 3 else adjusted[3]
            projected.append(self.input_feature_projection_convolution[i](inp))

        # Step 4: Bottom-up PAN (stride-2 conv for downsampling)
        bottom_up = [None] * 4
        bottom_up[0] = projected[0]
        for i in range(1, 4):
            bottom_up[i] = projected[i] + self.path_aggregation_head_convolution[i - 1](bottom_up[i - 1])

        # Step 5: Lateral refinement
        lateral = []
        for i in range(4):
            inp = projected[0] if i == 0 else bottom_up[i]
            lateral.append(self.path_aggregation_lateral_convolution[i](inp))

        # Step 6: Intra-class blocks
        refined = [self.intraclass_blocks[i](lateral[i]) for i in range(4)]

        # Step 7: Upsample all to largest size, concat reversed
        target_h, target_w = refined[0].shape[1], refined[0].shape[2]
        upsampled = []
        for feat in refined:
            upsampled.append(nearest_upsample_to(feat, target_h, target_w))

        return mx.concatenate(upsampled[::-1], axis=-1)
