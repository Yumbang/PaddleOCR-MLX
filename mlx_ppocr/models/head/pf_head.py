"""PFHeadLocal detection head (DB-based with local refinement) for PP-OCRv5 server."""

import mlx.core as mx
import mlx.nn as nn

from mlx_ppocr.utils.ops import nearest_upsample


class ConvBNRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int, padding: int = 0, bias: bool = False):
        super().__init__()
        self.convolution = nn.Conv2d(in_ch, out_ch, ks, padding=padding, bias=bias)
        self.norm = nn.BatchNorm(out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.norm(self.convolution(x)))


class LocalRefinementModule(nn.Module):
    """Refines binarization map using local features."""

    def __init__(self, in_channels: int):
        super().__init__()
        # Takes concatenation of features (in_channels) + binarized map (1)
        self.convolution_backbone = ConvBNRelu(in_channels + 1, in_channels, 3, padding=1)
        self.convolution_final = nn.Conv2d(in_channels, 1, 1, bias=True)

    def __call__(self, features: mx.array, binary_map: mx.array) -> mx.array:
        x = mx.concatenate([features, binary_map], axis=-1)
        x = self.convolution_backbone(x)
        x = self.convolution_final(x)
        return mx.sigmoid(x)


class PFHeadLocal(nn.Module):
    """PFHeadLocal: DB detection head with parallel fusion and local refinement.

    Input: fused feature map from neck (256 channels for mode="large" → in_channels//4=64)
    Output: probability map (N, H, W, 1) at original image scale
    """

    def __init__(self, in_channels: int = 256, mode: str = "large"):
        super().__init__()
        mid_channels = in_channels // 4 if mode == "large" else in_channels // 8

        # Binarize head: conv_down → conv_up (deconv) → conv_final (deconv)
        self.binarize_head = BinarizeHead(in_channels, mid_channels)

        # Local refinement module
        self.local_refinement_module = LocalRefinementModule(mid_channels)

    def __call__(self, x: mx.array) -> mx.array:
        features, binary_map = self.binarize_head(x)
        # Upsample features to match binary_map spatial size (2x)
        up_features = nearest_upsample(features, 2, 2)
        refined_map = self.local_refinement_module(up_features, binary_map)
        # Average base and refined maps during inference
        return (binary_map + refined_map) / 2.0


class BinarizeHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int):
        super().__init__()
        self.conv_down = ConvBNRelu(in_channels, mid_channels, 3, padding=1)
        self.conv_up = ConvTransposeBNRelu(mid_channels, mid_channels, 2, stride=2)
        self.conv_final = nn.ConvTranspose2d(mid_channels, 1, 2, stride=2, bias=True)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        x = self.conv_down(x)
        features = self.conv_up(x)
        binary_map = mx.sigmoid(self.conv_final(features))
        return features, binary_map


class ConvTransposeBNRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int, stride: int = 1):
        super().__init__()
        self.convolution = nn.ConvTranspose2d(in_ch, out_ch, ks, stride=stride, bias=True)
        self.norm = nn.BatchNorm(out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.norm(self.convolution(x)))
