"""Weight conversion from HuggingFace safetensors (PyTorch) to MLX."""

import argparse

import mlx.core as mx
import numpy as np
from safetensors import safe_open

# Keys that are ConvTranspose2d weights (PyTorch format: IOHW → MLX: OHWI)
_CONV_TRANSPOSE_KEYS = {
    "head.binarize_head.conv_up.convolution.weight",
    "head.binarize_head.conv_final.weight",
}


def convert_weights(src_path: str, dst_path: str, model_type: str = "det"):
    """Convert HF safetensors weights to MLX format.

    All 4D weights are transposed:
    - Conv2d: OIHW (PyTorch) → OHWI (MLX) via transpose(0,2,3,1)
    - ConvTranspose2d: IOHW (PyTorch) → OHWI (MLX) via transpose(1,2,3,0)
    """
    weights = {}
    with safe_open(src_path, framework="numpy") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)

            if "num_batches_tracked" in key:
                continue

            if tensor.ndim == 4 and key.endswith(".weight"):
                if model_type == "det" and key in _CONV_TRANSPOSE_KEYS:
                    # ConvTranspose2d: IOHW → OHWI
                    tensor = np.transpose(tensor, (1, 2, 3, 0))
                else:
                    # Conv2d: OIHW → OHWI
                    tensor = np.transpose(tensor, (0, 2, 3, 1))

            weights[key] = mx.array(tensor)

    mx.savez(dst_path, **weights)
    print(f"Converted {len(weights)} tensors → {dst_path}")
    return weights


def _set_nested_attr(obj, parts: list[str], value):
    """Set a nested attribute by path parts."""
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def convert_paddle_weights(pdiparams_path: str, dst_path: str):
    """Convert PaddlePaddle inference weights to MLX format.

    Requires paddlepaddle to be installed. Used for language-specific
    mobile rec models that don't have safetensors versions.
    """
    from pathlib import Path

    import paddle

    # Load parameters using paddle static graph
    paddle.enable_static()
    model_prefix = str(Path(pdiparams_path).with_suffix(""))

    import paddle.base as base

    place = base.CPUPlace()
    exe = base.Executor(place)
    program, _, _ = base.io.load_inference_model(model_prefix, exe)

    weights = {}
    for var in program.list_vars():
        if not var.persistable or var.name == "feed" or var.name == "fetch":
            continue
        try:
            tensor = np.array(base.global_scope().find_var(var.name).get_tensor())
        except Exception:
            continue

        if tensor.size == 0:
            continue

        # Map PaddlePaddle key to HF-style key
        key = var.name

        # Transpose Conv2d weights: OIHW (Paddle) → OHWI (MLX)
        if tensor.ndim == 4 and not key.endswith("_mean") and not key.endswith("_variance"):
            tensor = np.transpose(tensor, (0, 2, 3, 1))

        weights[key] = mx.array(tensor)

    paddle.disable_static()

    mx.savez(dst_path, **weights)
    print(f"Converted {len(weights)} tensors → {dst_path}")
    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Source safetensors file")
    parser.add_argument("dst", help="Destination .npz file")
    parser.add_argument("--type", choices=["det", "rec"], default="det")
    args = parser.parse_args()
    convert_weights(args.src, args.dst, args.type)
