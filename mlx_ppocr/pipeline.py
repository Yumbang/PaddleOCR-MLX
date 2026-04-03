"""End-to-end OCR pipeline: detect → crop → recognize."""

import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from mlx_ppocr.convert import convert_weights, _set_nested_attr
from mlx_ppocr.models.det_model import DetModel
from mlx_ppocr.models.rec_model import RecModel, MobileRecModel
from mlx_ppocr.processing.det_preprocess import det_preprocess
from mlx_ppocr.processing.det_postprocess import det_postprocess
from mlx_ppocr.processing.rec_preprocess import crop_text_region, rec_preprocess_crop
from mlx_ppocr.processing.rec_postprocess import ctc_decode, load_vocab

DET_HF_REPO = "PaddlePaddle/PP-OCRv5_server_det_safetensors"

# --- Model Registry ---

LANG_REGISTRY = {
    "server": {
        "rec_hf_repo": "PaddlePaddle/PP-OCRv5_server_rec_safetensors",
        "rec_cache_name": "server_rec",
        "model_variant": "server",
        "weight_format": "safetensors",
        "vocab_file": "preprocessor_config.json",
    },
    "mobile": {
        "rec_hf_repo": "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors",
        "rec_cache_name": "mobile_rec",
        "model_variant": "mobile",
        "weight_format": "safetensors",
        "vocab_file": "preprocessor_config.json",
    },
}

# Language-specific models (all mobile architecture, pdiparams format)
_PDIPARAMS_LANGS = {
    "korean": "PaddlePaddle/korean_PP-OCRv5_mobile_rec",
    "latin": "PaddlePaddle/latin_PP-OCRv5_mobile_rec",
    "cyrillic": "PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec",
    "arabic": "PaddlePaddle/arabic_PP-OCRv5_mobile_rec",
    "devanagari": "PaddlePaddle/devanagari_PP-OCRv5_mobile_rec",
    "thai": "PaddlePaddle/th_PP-OCRv5_mobile_rec",
    "greek": "PaddlePaddle/el_PP-OCRv5_mobile_rec",
    "tamil": "PaddlePaddle/ta_PP-OCRv5_mobile_rec",
    "telugu": "PaddlePaddle/te_PP-OCRv5_mobile_rec",
    "english": "PaddlePaddle/en_PP-OCRv5_mobile_rec",
    "eslav": "PaddlePaddle/eslav_PP-OCRv5_mobile_rec",
}
for _lang, _repo in _PDIPARAMS_LANGS.items():
    LANG_REGISTRY[_lang] = {
        "rec_hf_repo": _repo,
        "rec_cache_name": f"{_lang}_rec",
        "model_variant": "mobile",
        "weight_format": "pdiparams",
        "vocab_file": "config.json",
    }

# Convenience aliases
_ALIASES = {
    "japanese": "mobile",
    "chinese": "server",
    "spanish": "latin", "french": "latin", "german": "latin",
    "italian": "latin", "portuguese": "latin",
    "russian": "cyrillic",
    "hindi": "devanagari",
    "persian": "arabic",
}
for _alias, _target in _ALIASES.items():
    LANG_REGISTRY[_alias] = LANG_REGISTRY[_target]


class MLXOCR:
    """PP-OCRv5 OCR on Apple MLX with multi-language support.

    Usage:
        ocr = MLXOCR()                    # server (Chinese + English)
        ocr = MLXOCR(lang="mobile")       # mobile (Chinese + English + Japanese)
        ocr = MLXOCR(lang="korean")       # Korean + English
        ocr = MLXOCR(lang="latin")        # French, Spanish, German, etc.
        results = ocr("path/to/image.jpg")
        for box, text, score in results:
            print(f"{text} ({score:.2f})")
    """

    def __init__(
        self,
        lang: str = "server",
        det_weights: str | None = None,
        rec_weights: str | None = None,
        vocab_path: str | None = None,
        cache_dir: str = "weights",
    ):
        if lang not in LANG_REGISTRY:
            available = sorted(set(LANG_REGISTRY.keys()) - set(_ALIASES.keys()))
            raise ValueError(
                f"Unknown language '{lang}'. Available: {', '.join(available)}. "
                f"Aliases: {', '.join(f'{a}→{t}' for a, t in _ALIASES.items())}"
            )

        self.lang_config = LANG_REGISTRY[lang]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Detection model (always server — language-agnostic)
        det_mlx = self._ensure_det_weights(det_weights)
        print("Loading detection model...")
        self.det_model = DetModel()
        self._load_weights(self.det_model, det_mlx)
        self.det_model.eval()

        # Recognition model (varies by language)
        rec_mlx = self._ensure_rec_weights(rec_weights)
        vocab_path = self._ensure_vocab(vocab_path)
        self.vocab = load_vocab(vocab_path)
        vocab_size = len(self.vocab)

        variant = self.lang_config["model_variant"]
        print(f"Loading {variant} recognition model ({lang})...")
        if variant == "server":
            self.rec_model = RecModel(vocab_size=vocab_size)
        else:
            self.rec_model = MobileRecModel(vocab_size=vocab_size)
        self._load_weights(self.rec_model, rec_mlx)
        self.rec_model.eval()
        print(f"Loaded vocabulary: {vocab_size} characters")

    def __call__(
        self,
        image: str | np.ndarray | Image.Image,
        det_thresh: float = 0.1,
        box_thresh: float = 0.3,
        unclip_ratio: float = 1.5,
    ) -> list[tuple[np.ndarray, str, float]]:
        """Run OCR on an image.

        Args:
            image: File path, numpy array (HWC RGB), or PIL Image
            det_thresh: Detection binarization threshold
            box_thresh: Minimum box confidence
            unclip_ratio: Polygon expansion ratio

        Returns:
            List of (box_points, text, confidence) tuples
        """
        if isinstance(image, str):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        # Detection
        boxes = self._detect(image, det_thresh, box_thresh, unclip_ratio)

        # Crop all valid text regions
        crops = []
        crop_points = []
        for box_info in boxes:
            points = box_info["points"]

            w = max(np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[3] - points[2]))
            h = max(np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2]))
            if w < 5 or h < 5:
                continue

            crop = crop_text_region(image, points)
            if crop is None or crop.shape[0] < 3 or crop.shape[1] < 3:
                continue

            crops.append(crop)
            crop_points.append(points)

        if not crops:
            return []

        rec_results = self._recognize_batch(crops)

        results = []
        for points, (text, score) in zip(crop_points, rec_results):
            if text and score > 0.5:
                results.append((points, text, score))

        results.sort(key=lambda x: x[0][:, 1].min())
        return results

    def _detect(
        self, image: np.ndarray, thresh: float, box_thresh: float,
        unclip_ratio: float,
    ) -> list[dict]:
        preprocessed, meta = det_preprocess(image)
        x = mx.array(preprocessed)
        prob_map = self.det_model(x)
        mx.eval(prob_map)
        prob_map = np.array(prob_map[0, :, :, 0])

        return det_postprocess(
            prob_map,
            meta["src_h"], meta["src_w"],
            meta["resize_h"], meta["resize_w"],
            thresh=thresh,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
        )

    def _recognize_batch(
        self, crops: list[np.ndarray], rec_batch_num: int = 6,
    ) -> list[tuple[str, float]]:
        """Batch recognition (PaddleOCR-style: sort by width, batch process)."""
        width_ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
        indices = np.argsort(np.array(width_ratios))

        all_results: list[tuple[str, float] | None] = [None] * len(crops)
        for beg in range(0, len(crops), rec_batch_num):
            end = min(len(crops), beg + rec_batch_num)
            batch_indices = indices[beg:end]

            norm_imgs = [rec_preprocess_crop(crops[idx]) for idx in batch_indices]

            batch = np.stack(norm_imgs, axis=0)
            x = mx.array(batch)
            logits = self.rec_model(x)
            mx.eval(logits)
            logits = np.array(logits)

            decoded = ctc_decode(logits, self.vocab)
            for i, idx in enumerate(batch_indices):
                all_results[idx] = decoded[i]

        return all_results

    # --- Weight management ---

    def _ensure_det_weights(self, path: str | None) -> str:
        mlx_path = self.cache_dir / "server_det.npz"
        if path and Path(path).exists():
            if path.endswith(".npz"):
                return path
            convert_weights(path, str(mlx_path), "det")
            return str(mlx_path)

        if mlx_path.exists():
            return str(mlx_path)

        print(f"Downloading detection model from {DET_HF_REPO}...")
        sf_path = hf_hub_download(DET_HF_REPO, "model.safetensors")
        print("Converting detection weights to MLX format...")
        convert_weights(sf_path, str(mlx_path), "det")
        return str(mlx_path)

    def _ensure_rec_weights(self, path: str | None) -> str:
        cfg = self.lang_config
        cache_name = cfg["rec_cache_name"]
        mlx_path = self.cache_dir / f"{cache_name}.npz"

        if path and Path(path).exists():
            if path.endswith(".npz"):
                return path
            convert_weights(path, str(mlx_path), "rec")
            return str(mlx_path)

        if mlx_path.exists():
            return str(mlx_path)

        repo = cfg["rec_hf_repo"]
        fmt = cfg["weight_format"]

        if fmt == "safetensors":
            print(f"Downloading rec model from {repo}...")
            sf_path = hf_hub_download(repo, "model.safetensors")
            print("Converting rec weights to MLX format...")
            convert_weights(sf_path, str(mlx_path), "rec")
        elif fmt == "pdiparams":
            self._convert_pdiparams(repo, mlx_path)
        else:
            raise ValueError(f"Unknown weight format: {fmt}")

        return str(mlx_path)

    def _convert_pdiparams(self, repo: str, mlx_path: Path):
        """Convert PaddlePaddle .pdiparams weights to MLX .npz.

        Requires paddlepaddle to be installed.
        """
        try:
            import paddle  # noqa: F401
        except ImportError:
            raise ImportError(
                "This language model requires paddlepaddle for first-time "
                "weight conversion.\nInstall with:\n"
                "  uv pip install paddlepaddle\n"
                "  # or\n"
                "  uv tool install mlx-ppocr[multilingual]"
            ) from None

        from mlx_ppocr.convert import convert_paddle_weights

        print(f"Downloading rec model from {repo}...")
        pdiparams_path = hf_hub_download(repo, "inference.pdiparams")
        # Also need the model file for paddle to load parameters
        hf_hub_download(repo, "inference.pdiparams.info")
        print("Converting PaddlePaddle weights to MLX format...")
        convert_paddle_weights(pdiparams_path, str(mlx_path))

    def _ensure_vocab(self, path: str | None) -> str:
        if path and Path(path).exists():
            return path

        cfg = self.lang_config
        cache_name = cfg["rec_cache_name"]
        vocab_file = cfg["vocab_file"]
        vocab_path = self.cache_dir / f"{cache_name}_vocab.json"

        if vocab_path.exists():
            return str(vocab_path)

        repo = cfg["rec_hf_repo"]
        print(f"Downloading vocabulary from {repo}...")
        downloaded = hf_hub_download(repo, vocab_file)
        shutil.copy(downloaded, vocab_path)
        return str(vocab_path)

    def _load_weights(self, model, weights_path: str):
        weights = dict(mx.load(weights_path))
        loaded = 0
        missing = []
        for key, value in weights.items():
            parts = key.split(".")
            try:
                _set_nested_attr(model, parts, value)
                loaded += 1
            except (AttributeError, IndexError, KeyError, TypeError):
                missing.append(key)

        print(f"  Loaded {loaded}/{len(weights)} parameters")
        if missing:
            print(f"  {len(missing)} unmatched keys (first 5): {missing[:5]}")
