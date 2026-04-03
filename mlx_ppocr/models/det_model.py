"""Full detection model: PPHGNetV2 backbone + LKPAN neck + PFHeadLocal head."""

import mlx.core as mx
import mlx.nn as nn

from mlx_ppocr.models.backbone.pphgnetv2 import PPHGNetV2
from mlx_ppocr.models.neck.lkpan import LKPAN
from mlx_ppocr.models.head.pf_head import PFHeadLocal


class DetModel(nn.Module):
    """PP-OCRv5 server detection model."""

    def __init__(self):
        super().__init__()
        self.model = DetModelInner()
        self.head = PFHeadLocal(in_channels=256, mode="large")

    def __call__(self, x: mx.array) -> mx.array:
        features = self.model(x)
        return self.head(features)


class DetModelInner(nn.Module):
    """Backbone + neck wrapper matching HF weight key prefix 'model.'."""

    def __init__(self):
        super().__init__()
        self.backbone = PPHGNetV2(
            stage_downsample=[False, True, True, True],
            stem_strides=[2, 1, 1, 2, 1],
            stage_downsample_strides=[2, 2, 2, 2],
            act="relu",
        )
        self.neck = LKPAN(
            in_channels=[128, 512, 1024, 2048],
            neck_out_channels=256,
        )

    def __call__(self, x: mx.array) -> mx.array:
        features = self.backbone(x)
        return self.neck(features)
