"""Tests for mlx_ppocr.processing.det_preprocess."""

import numpy as np
from PIL import Image

from mlx_ppocr.processing.det_preprocess import det_preprocess


class TestDetPreprocess:
    def test_output_shape_and_batch_dim(self, rgb_image_100x200):
        result, meta = det_preprocess(rgb_image_100x200)
        assert result.ndim == 4
        assert result.shape[0] == 1
        assert result.shape[3] == 3

    def test_metadata_has_required_keys(self, rgb_image_100x200):
        _, meta = det_preprocess(rgb_image_100x200)
        assert set(meta.keys()) == {"src_h", "src_w", "resize_h", "resize_w", "ratio"}
        assert meta["src_h"] == 100
        assert meta["src_w"] == 200

    def test_resize_h_w_are_multiples_of_32(self, rgb_image_100x200):
        _, meta = det_preprocess(rgb_image_100x200)
        assert meta["resize_h"] % 32 == 0
        assert meta["resize_w"] % 32 == 0

    def test_small_image_gets_minimum_32(self):
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        result, meta = det_preprocess(tiny)
        assert meta["resize_h"] >= 32
        assert meta["resize_w"] >= 32

    def test_large_image_downscaled(self):
        big = np.zeros((2000, 3000, 3), dtype=np.uint8)
        _, meta = det_preprocess(big, limit_side_len=960)
        assert meta["resize_h"] <= 960
        assert meta["resize_w"] <= 960
        assert meta["ratio"] < 1.0

    def test_image_within_limit_not_resized(self):
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        _, meta = det_preprocess(img, limit_side_len=960)
        assert meta["ratio"] == 1.0
        assert meta["resize_h"] == 320
        assert meta["resize_w"] == 320

    def test_normalization_range(self, rgb_image_100x200):
        result, _ = det_preprocess(rgb_image_100x200)
        # ImageNet normalization: roughly in [-2.5, 2.8] range
        assert result.dtype == np.float32
        assert result.min() > -3.0
        assert result.max() < 3.5

    def test_pil_image_input(self):
        pil = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        result, meta = det_preprocess(pil)
        assert result.shape == (1, 64, 64, 3)
        assert meta["src_h"] == 64

    def test_aspect_ratio_preserved(self):
        img = np.zeros((100, 400, 3), dtype=np.uint8)
        _, meta = det_preprocess(img, limit_side_len=960)
        # Aspect ratio of resize should be close to 1:4
        actual_ratio = meta["resize_w"] / meta["resize_h"]
        assert 3.0 < actual_ratio < 5.0  # allow rounding tolerance

    def test_custom_limit_side_len(self):
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        _, meta = det_preprocess(img, limit_side_len=256)
        assert meta["resize_h"] <= 256
        assert meta["resize_w"] <= 256
