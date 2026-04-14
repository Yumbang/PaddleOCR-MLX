"""Tests for mlx_ppocr.processing.rec_preprocess."""

import numpy as np

from mlx_ppocr.processing.rec_preprocess import (
    crop_text_region,
    rec_preprocess,
    rec_preprocess_crop,
)


class TestCropTextRegion:
    def test_basic_crop(self, rgb_image_100x200):
        pts = np.array([[10, 10], [100, 10], [100, 40], [10, 40]], dtype=np.float32)
        crop = crop_text_region(rgb_image_100x200, pts)
        assert crop is not None
        assert crop.ndim == 3
        assert crop.shape[2] == 3  # RGB

    def test_returns_none_for_zero_size(self, rgb_image_100x200):
        pts = np.array([[10, 10], [10, 10], [10, 10], [10, 10]], dtype=np.float32)
        assert crop_text_region(rgb_image_100x200, pts) is None

    def test_vertical_text_rotated(self, rgb_image_100x200):
        # Tall narrow box: h/w >= 1.5 → should be rotated
        pts = np.array([[50, 0], [60, 0], [60, 90], [50, 90]], dtype=np.float32)
        crop = crop_text_region(rgb_image_100x200, pts)
        assert crop is not None
        # After rotation, width > height
        assert crop.shape[1] >= crop.shape[0]

    def test_horizontal_text_not_rotated(self, rgb_image_100x200):
        pts = np.array([[10, 40], [150, 40], [150, 60], [10, 60]], dtype=np.float32)
        crop = crop_text_region(rgb_image_100x200, pts)
        assert crop is not None
        # Width should be much larger than height
        assert crop.shape[1] > crop.shape[0]

    def test_float_coordinates(self, rgb_image_100x200):
        pts = np.array(
            [[10.5, 10.3], [99.7, 11.1], [99.2, 39.8], [10.1, 39.2]],
            dtype=np.float32,
        )
        crop = crop_text_region(rgb_image_100x200, pts)
        assert crop is not None


class TestRecPreprocessCrop:
    def test_output_shape(self):
        crop = np.random.randint(0, 256, (30, 100, 3), dtype=np.uint8)
        result = rec_preprocess_crop(crop, target_height=48, max_width=320)
        assert result.shape == (48, 320, 3)

    def test_output_is_float32(self):
        crop = np.random.randint(0, 256, (30, 100, 3), dtype=np.uint8)
        result = rec_preprocess_crop(crop)
        assert result.dtype == np.float32

    def test_normalization_range(self):
        crop = np.random.randint(0, 256, (30, 100, 3), dtype=np.uint8)
        result = rec_preprocess_crop(crop)
        # (x/255 - 0.5)/0.5 → range [-1, 1] for actual content, 0 for padding
        assert result.min() >= -1.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_padding_is_zero(self):
        crop = np.full((30, 50, 3), 128, dtype=np.uint8)
        result = rec_preprocess_crop(crop, target_height=48, max_width=320)
        # The right part should be zero-padded
        # Width after resize: 50 * (48/30) = 80
        # Padding from 80 to 320 should be zeros
        assert np.all(result[:, 100:, :] == 0.0)

    def test_wide_image_clipped_to_max_width(self):
        crop = np.random.randint(0, 256, (10, 1000, 3), dtype=np.uint8)
        result = rec_preprocess_crop(crop, target_height=48, max_width=320)
        assert result.shape == (48, 320, 3)

    def test_very_narrow_image(self):
        crop = np.random.randint(0, 256, (48, 2, 3), dtype=np.uint8)
        result = rec_preprocess_crop(crop, target_height=48, max_width=320)
        assert result.shape == (48, 320, 3)


class TestRecPreprocess:
    def test_adds_batch_dimension(self):
        crop = np.random.randint(0, 256, (30, 100, 3), dtype=np.uint8)
        result = rec_preprocess(crop, target_height=48, target_width=320)
        assert result.shape == (1, 48, 320, 3)
