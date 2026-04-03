"""SVTR encoder for PP-OCRv5 recognition (server and mobile)."""

import mlx.core as mx
import mlx.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int | tuple, act: str = "silu"):
        super().__init__()
        if isinstance(kernel_size, (list, tuple)):
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            padding = kernel_size // 2
        self.convolution = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.normalization = nn.BatchNorm(out_ch)
        self.act = act

    def __call__(self, x: mx.array) -> mx.array:
        x = self.convolution(x)
        x = self.normalization(x)
        if self.act == "silu":
            x = nn.silu(x)
        elif self.act == "relu":
            x = nn.relu(x)
        return x


class SVTRAttention(nn.Module):
    """Multi-head self-attention with fused QKV."""

    def __init__(self, hidden_size: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, C)
        return self.projection(x)


class SVTRMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, act: str = "silu"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = act

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        if self.act == "silu":
            x = nn.silu(x)
        elif self.act == "relu":
            x = nn.relu(x)
        x = self.fc2(x)
        return x


class SVTRBlock(nn.Module):
    """Single SVTR transformer block: LN → Attention → LN → MLP."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 2.0, act: str = "silu"):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = SVTRAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.mlp = SVTRMlp(hidden_size, int(hidden_size * mlp_ratio), act=act)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SVTREncoder(nn.Module):
    """SVTR encoder: conv blocks + transformer blocks for sequence modeling.

    Input: feature map from backbone [B, H, W, C]
           Server: C=2048, dims=256. Mobile: C=480, dims=60.
    Output: sequence features [B, W, hidden_size] where hidden_size=120
    """

    def __init__(
        self,
        in_channels: int = 2048,
        dims: int = 256,
        hidden_size: int = 120,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        conv_kernel_size: list[int] = [1, 3],
        act: str = "silu",
    ):
        super().__init__()

        # Conv blocks for channel manipulation
        self.conv_block = [
            ConvBNAct(in_channels, dims, (conv_kernel_size[0], conv_kernel_size[1]), act=act),
            ConvBNAct(dims, hidden_size, 1, act=act),
            ConvBNAct(hidden_size, in_channels, 1, act=act),
            ConvBNAct(in_channels * 2, dims, (conv_kernel_size[0], conv_kernel_size[1]), act=act),
            ConvBNAct(dims, hidden_size, 1, act=act),
        ]

        # Transformer blocks
        self.svtr_block = [
            SVTRBlock(hidden_size, num_heads, mlp_ratio, act=act)
            for _ in range(depth)
        ]

        self.norm = nn.LayerNorm(hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x_input = x  # [B, H, W, 2048]

        # Channel reduction path
        z = self.conv_block[0](x_input)    # [B, H, W, 256]
        h = self.conv_block[1](z)          # [B, H, W, 120]

        # Flatten spatial dims for transformer
        B, H, W, C = h.shape
        h = h.reshape(B, H * W, C)

        # Transformer blocks
        for block in self.svtr_block:
            h = block(h)
        h = self.norm(h)

        # Reshape back to spatial
        h = h.reshape(B, H, W, C)

        # Skip connection path
        h = self.conv_block[2](h)           # [B, H, W, 2048]
        h = mx.concatenate([x_input, h], axis=-1)  # [B, H, W, 4096]
        h = self.conv_block[3](h)           # [B, H, W, 256]
        h = self.conv_block[4](h)           # [B, H, W, 120]

        # Squeeze height (should be 1 after AvgPool in RecModelInner)
        h = h.squeeze(1)                    # [B, W, 120]
        return h
