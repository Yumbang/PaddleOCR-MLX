import mlx.core as mx


def hard_sigmoid(x: mx.array) -> mx.array:
    return mx.clip(0.2 * x + 0.5, 0.0, 1.0)


def hardswish(x: mx.array) -> mx.array:
    return x * hard_sigmoid(x)


def nearest_upsample(x: mx.array, scale_h: int, scale_w: int) -> mx.array:
    """Nearest-neighbor upsample for NHWC tensor."""
    N, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (N, H, scale_h, W, scale_w, C))
    return x.reshape(N, H * scale_h, W * scale_w, C)


def nearest_upsample_to(x: mx.array, target_h: int, target_w: int) -> mx.array:
    """Nearest-neighbor upsample NHWC tensor to target spatial size."""
    N, H, W, C = x.shape
    if H == target_h and W == target_w:
        return x
    row_idx = mx.arange(target_h) * H // target_h
    col_idx = mx.arange(target_w) * W // target_w
    return x[:, row_idx][:, :, col_idx]
