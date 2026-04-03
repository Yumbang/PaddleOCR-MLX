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


def convert_paddle_weights(inference_json: str, pdiparams: str, dst_path: str):
    """Convert PaddlePaddle inference weights to MLX format.

    The inference model has BN fused into Conv (reparameterized).
    Uses position-based mapping built from the base mobile model
    (which has both pdiparams and safetensors formats available).

    Non-fused params (lab, act, identity BN, SE, SVTR, CTC) are mapped
    by position to HF keys. Fused conv+BN params are loaded into
    conv_symmetric[0] with identity BN.

    Requires paddlepaddle to be installed.
    """
    import os
    import shutil
    import tempfile

    import paddle
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    paddle.disable_static()

    # Step 1: Build position→HF_key mapping from base mobile model
    print("  Building parameter mapping from base model...")
    base_pdi = hf_hub_download("PaddlePaddle/PP-OCRv5_mobile_rec", "inference.pdiparams")
    base_jsn = hf_hub_download("PaddlePaddle/PP-OCRv5_mobile_rec", "inference.json")
    base_sf = hf_hub_download(
        "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors", "model.safetensors"
    )

    # Load base model paddle params
    tmpdir = tempfile.mkdtemp()
    shutil.copy(base_jsn, os.path.join(tmpdir, "inference.json"))
    shutil.copy(base_pdi, os.path.join(tmpdir, "inference.pdiparams"))
    base_pd = paddle.jit.load(os.path.join(tmpdir, "inference"))
    base_sd = base_pd.state_dict()
    base_keys = sorted(base_sd.keys(), key=lambda k: int(k.split("_")[1]))
    shutil.rmtree(tmpdir)

    # Load base model HF params
    with safe_open(base_sf, framework="numpy") as f:
        hf_map = {k: f.get_tensor(k) for k in sorted(f.keys())}

    # Match by value (direct + transposed) to build position mapping
    pos_mapping = {}  # paddle param index → (hf_key, needs_transpose)
    used_hf = set()
    for idx, pd_key in enumerate(base_keys):
        pd_val = base_sd[pd_key].numpy()
        best_match = None
        best_diff = float("inf")
        best_transposed = False

        for hf_key, hf_val in hf_map.items():
            if hf_key in used_hf:
                continue
            if pd_val.shape == hf_val.shape:
                diff = float(np.abs(pd_val - hf_val).max())
                if diff < best_diff:
                    best_diff = diff
                    best_match = hf_key
                    best_transposed = False
            if pd_val.ndim == 2 and hf_val.ndim == 2:
                if pd_val.shape == (hf_val.shape[1], hf_val.shape[0]):
                    diff = float(np.abs(pd_val - hf_val.T).max())
                    if diff < best_diff:
                        best_diff = diff
                        best_match = hf_key
                        best_transposed = True

        if best_match is not None and best_diff < 1e-4:
            pos_mapping[idx] = (best_match, best_transposed)
            used_hf.add(best_match)

    # Step 2: Load target model and apply mapping
    print("  Loading target model weights...")
    tmpdir2 = tempfile.mkdtemp()
    shutil.copy(inference_json, os.path.join(tmpdir2, "inference.json"))
    shutil.copy(pdiparams, os.path.join(tmpdir2, "inference.pdiparams"))
    tgt_pd = paddle.jit.load(os.path.join(tmpdir2, "inference"))
    tgt_sd = tgt_pd.state_dict()
    tgt_keys = sorted(tgt_sd.keys(), key=lambda k: int(k.split("_")[1]))
    shutil.rmtree(tmpdir2)

    weights = {}
    fused_keys = []

    for idx, pd_key in enumerate(tgt_keys):
        pd_val = tgt_sd[pd_key].numpy()

        if idx in pos_mapping:
            hf_key, transposed = pos_mapping[idx]
            tensor = pd_val.T if transposed else pd_val
            # Transpose 4D conv weights: OIHW → OHWI
            if tensor.ndim == 4:
                tensor = np.transpose(tensor, (0, 2, 3, 1))
            weights[hf_key] = mx.array(tensor)
        else:
            fused_keys.append(pd_key)

    # Step 3: Handle fused conv+BN params
    _load_fused_params(fused_keys, tgt_sd, weights)

    mx.savez(dst_path, **weights)
    print(f"  Converted {len(weights)} tensors → {dst_path}")
    return weights


def _load_fused_params(fused_keys, pd_sd, weights):
    """Map fused (bias, weight) pairs to LearnableRepLayer branches.

    The constant_folding pass reparameterizes the multi-branch
    LearnableRepLayer into a single conv. We load the fused weight
    into conv_symmetric[0] with identity BN and zero out other branches.
    """
    from mlx_ppocr.models.backbone.pplcnetv3 import LCNETV3_REC_BLOCK_CONFIGS

    # Build ordered list of (block_idx, layer_idx, "dw"/"pw") positions
    positions = []
    for bi, stage in enumerate(LCNETV3_REC_BLOCK_CONFIGS):
        for li in range(len(stage)):
            positions.append((bi, li, "dw"))
            positions.append((bi, li, "pw"))

    # Each position gets 2 fused params: (bias, weight)
    fused_idx = 0
    for bi, li, conv_type in positions:
        if fused_idx + 1 >= len(fused_keys):
            break

        bias_key = fused_keys[fused_idx]
        weight_key = fused_keys[fused_idx + 1]
        fused_bias = pd_sd[bias_key].numpy()
        fused_weight = pd_sd[weight_key].numpy()  # OIHW

        if fused_weight.ndim != 4:
            continue

        fused_idx += 2

        conv_name = (
            "depthwise_convolution" if conv_type == "dw"
            else "pointwise_convolution"
        )
        prefix = f"model.backbone.encoder.blocks.{bi}.layers.{li}.{conv_name}"
        out_ch = fused_weight.shape[0]
        fused_weight_ohwi = np.transpose(fused_weight, (0, 2, 3, 1))

        # Set conv_symmetric[0] to fused weight with BN carrying the fused bias
        weights[f"{prefix}.conv_symmetric.0.convolution.weight"] = mx.array(fused_weight_ohwi)
        _set_identity_bn(weights, f"{prefix}.conv_symmetric.0.normalization", out_ch)
        # Put fused bias into BN bias (identity BN: output = conv_out + bias)
        weights[f"{prefix}.conv_symmetric.0.normalization.bias"] = mx.array(fused_bias)

        # Zero out remaining conv_symmetric branches
        for branch_idx in range(1, 4):
            weights[f"{prefix}.conv_symmetric.{branch_idx}.convolution.weight"] = (
                mx.zeros_like(mx.array(fused_weight_ohwi))
            )
            _set_identity_bn(weights, f"{prefix}.conv_symmetric.{branch_idx}.normalization", out_ch)

        # Zero out conv_small_symmetric (exists for depthwise where kernel > 1)
        if conv_type == "dw":
            weights[f"{prefix}.conv_small_symmetric.convolution.weight"] = mx.zeros(
                (out_ch, 1, 1, fused_weight.shape[1])
            )
            _set_identity_bn(weights, f"{prefix}.conv_small_symmetric.normalization", out_ch)

        # Zero out identity BN if it exists and hasn't been set by mapped params
        id_key = f"{prefix}.identity.weight"
        if id_key not in weights:
            # Check if identity should exist (in_ch==out_ch and stride==1)
            # Just set it defensively; if it doesn't exist in the model, it'll be ignored
            weights[f"{prefix}.identity.weight"] = mx.zeros((out_ch,))
            weights[f"{prefix}.identity.bias"] = mx.zeros((out_ch,))
            weights[f"{prefix}.identity.running_mean"] = mx.zeros((out_ch,))
            weights[f"{prefix}.identity.running_var"] = mx.ones((out_ch,))

        # The fused_bias needs to be incorporated. Since lab (already mapped)
        # applies scale*x + bias AFTER summing branches, and the fused model
        # has bias baked into the conv, we set a separate bias through lab
        # if lab wasn't mapped. Otherwise fused_bias is already handled.
        lab_scale_key = f"{prefix}.lab.scale"
        if lab_scale_key not in weights:
            weights[lab_scale_key] = mx.ones((1,))
            weights[f"{prefix}.lab.bias"] = mx.zeros((1,))


def _set_identity_bn(weights, prefix, channels):
    """Set BatchNorm to identity transform."""
    weights[f"{prefix}.weight"] = mx.ones((channels,))
    weights[f"{prefix}.bias"] = mx.zeros((channels,))
    weights[f"{prefix}.running_mean"] = mx.zeros((channels,))
    weights[f"{prefix}.running_var"] = mx.ones((channels,))


def _is_linear_weight(key: str) -> bool:
    """Check if a key corresponds to a Linear layer weight (needs transpose)."""
    return (
        "qkv.weight" in key
        or "proj.weight" in key
        or "fc1.weight" in key
        or "fc2.weight" in key
        or "fc.weight" in key
    )


# PaddleOCR block numbering → HF block index
_BLOCK_MAP = {
    "blocks2": (0, 0), "blocks3": (1, None), "blocks4": (2, None),
    "blocks5": (3, None), "blocks6": (4, None),
}


def _map_paddle_key(pd_key: str) -> str | None:
    """Map PaddleOCR param key to HF-style key for MLX model."""
    # Skip GTC (attention-based) head — we only use CTC
    if "gtc" in pd_key or "before_gtc" in pd_key:
        return None

    # Stem conv
    if pd_key.startswith("backbone.conv1."):
        rest = pd_key[len("backbone.conv1."):]
        rest = _map_conv_bn(rest)
        return f"model.backbone.encoder.convolution.{rest}"

    # Backbone blocks
    for block_name, (block_idx, _) in _BLOCK_MAP.items():
        prefix = f"backbone.{block_name}."
        if pd_key.startswith(prefix):
            rest = pd_key[len(prefix):]
            # Extract layer index
            layer_idx = int(rest.split(".")[0])
            rest = rest[len(str(layer_idx)) + 1:]
            return _map_layer_key(block_idx, layer_idx, rest)

    # SVTR encoder
    if pd_key.startswith("head.ctc_encoder.encoder."):
        rest = pd_key[len("head.ctc_encoder.encoder."):]
        return _map_svtr_key(rest)

    # CTC head
    if pd_key.startswith("head.ctc_head.fc."):
        rest = pd_key[len("head.ctc_head.fc."):]
        return f"head.head.{rest}"

    return None


def _map_layer_key(block_idx: int, layer_idx: int, rest: str) -> str:
    """Map a single layer's key within a block."""
    prefix = f"model.backbone.encoder.blocks.{block_idx}.layers.{layer_idx}"

    if rest.startswith("dw_conv."):
        sub = rest[len("dw_conv."):]
        return f"{prefix}.depthwise_convolution.{_map_rep_layer(sub)}"
    elif rest.startswith("pw_conv."):
        sub = rest[len("pw_conv."):]
        return f"{prefix}.pointwise_convolution.{_map_rep_layer(sub)}"
    elif rest.startswith("se."):
        sub = rest[len("se."):]
        return f"{prefix}.squeeze_excitation_module.{_map_se(sub)}"

    return f"{prefix}.{rest}"


def _map_rep_layer(key: str) -> str:
    """Map LearnableRepLayer sub-keys."""
    # conv_kxk.{i}.conv.weight → conv_symmetric.{i}.convolution.weight
    if key.startswith("conv_kxk."):
        rest = key[len("conv_kxk."):]
        idx = rest.split(".")[0]
        sub = rest[len(idx) + 1:]
        sub = _map_conv_bn(sub)
        return f"conv_symmetric.{idx}.{sub}"

    # conv_1x1.conv.weight → conv_small_symmetric.convolution.weight
    if key.startswith("conv_1x1."):
        rest = key[len("conv_1x1."):]
        rest = _map_conv_bn(rest)
        return f"conv_small_symmetric.{rest}"

    # bn_branch → identity
    if key.startswith("bn_branch."):
        rest = key[len("bn_branch."):]
        rest = _map_bn(rest)
        return f"identity.{rest}"

    # lab.scale, lab.bias → lab.scale, lab.bias
    if key.startswith("lab.") or key.startswith("act."):
        return key

    return key


def _map_se(key: str) -> str:
    """Map SE module keys."""
    # se.conv1 → convolutions.0
    # se.conv2 → convolutions.2
    if key.startswith("conv1."):
        rest = key[len("conv1."):]
        return f"convolutions.0.{rest}"
    if key.startswith("conv2."):
        rest = key[len("conv2."):]
        return f"convolutions.2.{rest}"
    return key


def _map_conv_bn(key: str) -> str:
    """Map conv/bn sub-keys to HF format."""
    return (
        key.replace("conv.weight", "convolution.weight")
        .replace("bn.weight", "normalization.weight")
        .replace("bn.bias", "normalization.bias")
        .replace("bn._mean", "normalization.running_mean")
        .replace("bn._variance", "normalization.running_var")
        .replace("norm.weight", "normalization.weight")
        .replace("norm.bias", "normalization.bias")
        .replace("norm._mean", "normalization.running_mean")
        .replace("norm._variance", "normalization.running_var")
    )


def _map_bn(key: str) -> str:
    """Map standalone BN keys."""
    return (
        key.replace("_mean", "running_mean")
        .replace("_variance", "running_var")
    )


def _map_svtr_key(key: str) -> str:
    """Map SVTR encoder keys."""
    # Conv blocks: conv1→0, conv1x1→1, conv3→2, conv4→3, conv2→4
    conv_map = {"conv1.": "0.", "conv1x1.": "1.", "conv3.": "2.", "conv4.": "3.", "conv2.": "4."}
    for pd_name, hf_idx in conv_map.items():
        if key.startswith(pd_name):
            rest = key[len(pd_name):]
            rest = _map_conv_bn(rest)
            return f"head.encoder.conv_block.{hf_idx}{rest}"

    # SVTR transformer blocks
    if key.startswith("svtr_block."):
        rest = key[len("svtr_block."):]
        rest = (
            rest.replace("mixer.qkv.", "self_attn.qkv.")
            .replace("mixer.proj.", "self_attn.projection.")
            .replace("norm1.", "layer_norm1.")
            .replace("norm2.", "layer_norm2.")
        )
        return f"head.encoder.svtr_block.{rest}"

    # LayerNorm
    if key.startswith("norm."):
        return f"head.encoder.norm.{key[5:]}"

    return f"head.encoder.{key}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Source safetensors file")
    parser.add_argument("dst", help="Destination .npz file")
    parser.add_argument("--type", choices=["det", "rec"], default="det")
    args = parser.parse_args()
    convert_weights(args.src, args.dst, args.type)
