"""Full recognition model: PPHGNetV2 backbone + SVTR encoder + CTC head."""

import mlx.core as mx
import mlx.nn as nn

from mlx_ppocr.models.backbone.pphgnetv2 import PPHGNetV2
from mlx_ppocr.models.encoder.svtr import SVTREncoder
from mlx_ppocr.models.head.ctc_head import CTCHead


class RecModel(nn.Module):
    """PP-OCRv5 server recognition model."""

    def __init__(self):
        super().__init__()
        self.model = RecModelInner()
        self.head = RecHead()

    def __call__(self, x: mx.array) -> mx.array:
        features = self.model(x)
        return self.head(features)


class RecModelInner(nn.Module):
    """Backbone wrapper matching HF weight key prefix 'model.'."""

    def __init__(self):
        super().__init__()
        self.backbone = PPHGNetV2(
            stage_downsample=[True, True, True, True],
            stem_strides=[2, 1, 1, 1, 1],
            stage_downsample_strides=[[2, 1], [1, 2], [2, 1], [2, 1]],
            act="relu",
        )

    def __call__(self, x: mx.array) -> mx.array:
        features = self.backbone(x)
        x = features[-1]  # [B, H, W, 2048], H=3, W varies
        # AvgPool2d(kernel=(3, 2)) to reduce H→1, W→W/2
        # MLX AvgPool2d expects NHWC input
        x = nn.AvgPool2d(kernel_size=(3, 2))(x)  # [B, 1, W/2, 2048]
        return x


class RecHead(nn.Module):
    """SVTR encoder + CTC head matching HF weight key prefix 'head.'."""

    def __init__(self):
        super().__init__()
        self.encoder = SVTREncoder(
            in_channels=2048,
            dims=256,
            hidden_size=120,
            depth=2,
            num_heads=8,
            mlp_ratio=2.0,
            conv_kernel_size=[1, 3],
            act="silu",
        )
        self.head = CTCHead(in_channels=120, out_channels=18385)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.encoder(x)   # [B, W, 120]
        x = self.head(x)      # [B, W, 18385]
        return x
