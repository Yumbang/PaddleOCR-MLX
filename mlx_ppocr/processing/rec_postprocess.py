"""CTC decoding: logits → text strings."""

import json
from pathlib import Path

import numpy as np


def load_vocab(vocab_path: str) -> list[str]:
    """Load character vocabulary from file.

    Supports:
    - HF preprocessor_config.json (key: 'character_list')
    - PaddleOCR config.json (key: 'PostProcess.character_dict')
    - Plain text file with one character per line
    """
    path = Path(vocab_path)
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        # HF format: top-level character_list
        if "character_list" in data:
            return data["character_list"]
        # HF format: top-level character_dict
        if "character_dict" in data:
            return data["character_dict"]
        # PaddleOCR format: PostProcess.character_dict
        # Prepend blank token (index 0) since PaddleOCR dicts don't include it
        post = data.get("PostProcess", {})
        if "character_dict" in post:
            return ["blank"] + post["character_dict"]
        raise ValueError(f"No character list found in {vocab_path}")
    else:
        with open(path, encoding="utf-8") as f:
            return [line.strip("\n") for line in f]


def ctc_decode(
    logits: np.ndarray,
    vocab: list[str],
    blank_idx: int = 0,
) -> list[tuple[str, float]]:
    """Decode CTC logits to text.

    Args:
        logits: [B, T, vocab_size] raw logits
        vocab: Character list (index 0 = blank/padding)
        blank_idx: Index of blank token

    Returns:
        List of (text, confidence) tuples
    """
    # Softmax
    probs = _softmax(logits)
    # Argmax
    pred_ids = np.argmax(probs, axis=-1)  # [B, T]
    pred_scores = np.max(probs, axis=-1)  # [B, T]

    results = []
    for b in range(pred_ids.shape[0]):
        chars = []
        scores = []
        prev = blank_idx
        for t in range(pred_ids.shape[1]):
            idx = int(pred_ids[b, t])
            if idx != blank_idx and idx != prev:
                if idx < len(vocab):
                    chars.append(vocab[idx])
                    scores.append(float(pred_scores[b, t]))
            prev = idx

        text = "".join(chars)
        conf = float(np.mean(scores)) if scores else 0.0
        results.append((text, conf))

    return results


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)
