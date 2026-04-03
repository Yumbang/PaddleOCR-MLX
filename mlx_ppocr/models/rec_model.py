"""Recognition models: server (PPHGNetV2) and mobile (PP-LCNetV3)."""

import mlx.core as mx
import mlx.nn as nn

from mlx_ppocr.models.backbone.pphgnetv2 import PPHGNetV2
from mlx_ppocr.models.backbone.pplcnetv3 import PPLCNetV3
from mlx_ppocr.models.encoder.svtr import SVTREncoder
from mlx_ppocr.models.head.ctc_head import CTCHead


# --- Shared head ---

class RecHead(nn.Module):
    """SVTR encoder + CTC head matching HF weight key prefix 'head.'."""

    def __init__(self, in_channels: int = 2048, vocab_size: int = 18385):
        super().__init__()
        dims = in_channels // 8
        self.encoder = SVTREncoder(
            in_channels=in_channels,
            dims=dims,
            hidden_size=120,
            depth=2,
            num_heads=8,
            mlp_ratio=2.0,
            conv_kernel_size=[1, 3],
            act="silu",
        )
        self.head = CTCHead(in_channels=120, out_channels=vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.encoder(x)   # [B, W, 120]
        x = self.head(x)      # [B, W, vocab_size]
        return x


# --- Server recognition model ---

class RecModel(nn.Module):
    """PP-OCRv5 server recognition model (PPHGNetV2 + SVTR + CTC)."""

    def __init__(self, vocab_size: int = 18385):
        super().__init__()
        self.model = ServerRecModelInner()
        self.head = RecHead(in_channels=2048, vocab_size=vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        features = self.model(x)
        return self.head(features)


class ServerRecModelInner(nn.Module):
    """PPHGNetV2 backbone + AvgPool. Weight prefix: 'model.'."""

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
        x = features[-1]  # [B, H, W, 2048], H=3
        x = nn.AvgPool2d(kernel_size=(3, 2))(x)  # [B, 1, W/2, 2048]
        return x


# --- Mobile recognition model ---

class MobileRecModel(nn.Module):
    """PP-OCRv5 mobile recognition model (PP-LCNetV3 + SVTR + CTC)."""

    def __init__(self, vocab_size: int = 18385):
        super().__init__()
        self.model = MobileRecModelInner()
        self.head = RecHead(in_channels=480, vocab_size=vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        features = self.model(x)
        return self.head(features)


class MobileRecModelInner(nn.Module):
    """PP-LCNetV3 backbone + AvgPool. Weight prefix: 'model.'."""

    def __init__(self):
        super().__init__()
        self.backbone = PPLCNetV3()

    def __call__(self, x: mx.array) -> mx.array:
        x = self.backbone(x)  # [B, 3, 80, 480]
        x = nn.AvgPool2d(kernel_size=(3, 2))(x)  # [B, 1, 40, 480]
        return x
