"""End-to-end OCR pipeline: detect → crop → recognize."""

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from mlx_ppocr.convert import convert_weights, _set_nested_attr
from mlx_ppocr.models.det_model import DetModel
from mlx_ppocr.models.rec_model import RecModel
from mlx_ppocr.processing.det_preprocess import det_preprocess
from mlx_ppocr.processing.det_postprocess import det_postprocess
from mlx_ppocr.processing.rec_preprocess import crop_text_region, rec_preprocess_crop
from mlx_ppocr.processing.rec_postprocess import ctc_decode, load_vocab

DET_HF_REPO = "PaddlePaddle/PP-OCRv5_server_det_safetensors"
REC_HF_REPO = "PaddlePaddle/PP-OCRv5_server_rec_safetensors"


class MLXOCR:
    """PP-OCRv5 server OCR on Apple MLX.

    Usage:
        ocr = MLXOCR()
        results = ocr("path/to/image.jpg")
        for box, text, score in results:
            print(f"{text} ({score:.2f})")
    """

    def __init__(
        self,
        det_weights: str | None = None,
        rec_weights: str | None = None,
        vocab_path: str | None = None,
        cache_dir: str = "weights",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Download and convert weights if needed
        det_mlx = self._ensure_weights(det_weights, "det")
        rec_mlx = self._ensure_weights(rec_weights, "rec")
        vocab_path = self._ensure_vocab(vocab_path)

        # Load models
        print("Loading detection model...")
        self.det_model = DetModel()
        self._load_weights(self.det_model, det_mlx)
        self.det_model.eval()

        print("Loading recognition model...")
        self.rec_model = RecModel()
        self._load_weights(self.rec_model, rec_mlx)
        self.rec_model.eval()

        # Load vocabulary
        self.vocab = load_vocab(vocab_path)
        print(f"Loaded vocabulary: {len(self.vocab)} characters")

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

            # Filter out tiny boxes
            w = max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[3] - points[2]))
            h = max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
            if w < 5 or h < 5:
                continue

            crop = crop_text_region(image, points)
            if crop is None or crop.shape[0] < 3 or crop.shape[1] < 3:
                continue

            crops.append(crop)
            crop_points.append(points)

        if not crops:
            return []

        # Batch recognition (PaddleOCR-style: sort by width, process in batches)
        rec_results = self._recognize_batch(crops)

        results = []
        for points, (text, score) in zip(crop_points, rec_results):
            if text and score > 0.5:
                results.append((points, text, score))

        # Sort by vertical position (top to bottom)
        results.sort(key=lambda x: x[0][:, 1].min())
        return results

    def _detect(
        self, image: np.ndarray, thresh: float, box_thresh: float, unclip_ratio: float
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
        self, crops: list[np.ndarray], rec_batch_num: int = 6
    ) -> list[tuple[str, float]]:
        """Batch recognition ported from PaddleOCR.

        Sorts crops by width ratio, processes in batches of rec_batch_num,
        then reorders results back to original order.
        """
        # Sort by aspect ratio (width/height) for efficient batching
        width_ratios = [c.shape[1] / float(c.shape[0]) for c in crops]
        indices = np.argsort(np.array(width_ratios))

        # Process in batches
        all_results: list[tuple[str, float] | None] = [None] * len(crops)
        for beg in range(0, len(crops), rec_batch_num):
            end = min(len(crops), beg + rec_batch_num)
            batch_indices = indices[beg:end]

            # Preprocess each crop in this batch
            norm_imgs = []
            for idx in batch_indices:
                norm_img = rec_preprocess_crop(crops[idx])
                norm_imgs.append(norm_img)

            # Stack into batch tensor [B, H, W, 3]
            batch = np.stack(norm_imgs, axis=0)
            x = mx.array(batch)
            logits = self.rec_model(x)
            mx.eval(logits)
            logits = np.array(logits)

            decoded = ctc_decode(logits, self.vocab)
            for i, idx in enumerate(batch_indices):
                all_results[idx] = decoded[i]

        return all_results

    def _ensure_weights(self, path: str | None, model_type: str) -> str:
        mlx_path = self.cache_dir / f"server_{model_type}.npz"
        if path and Path(path).exists():
            if path.endswith(".npz"):
                return path
            convert_weights(path, str(mlx_path), model_type)
            return str(mlx_path)

        if mlx_path.exists():
            return str(mlx_path)

        # Download from HuggingFace
        repo = DET_HF_REPO if model_type == "det" else REC_HF_REPO
        print(f"Downloading {model_type} model from {repo}...")
        sf_path = hf_hub_download(repo, "model.safetensors")
        print(f"Converting {model_type} weights to MLX format...")
        convert_weights(sf_path, str(mlx_path), model_type)
        return str(mlx_path)

    def _ensure_vocab(self, path: str | None) -> str:
        if path and Path(path).exists():
            return path

        vocab_path = self.cache_dir / "preprocessor_config.json"
        if vocab_path.exists():
            return str(vocab_path)

        print("Downloading vocabulary...")
        downloaded = hf_hub_download(REC_HF_REPO, "preprocessor_config.json")
        import shutil
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
