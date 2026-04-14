"""Shared fixtures for mlx_ppocr tests."""

import numpy as np
import pytest


@pytest.fixture
def rgb_image_100x200():
    """100-high, 200-wide RGB image with non-trivial content."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (100, 200, 3), dtype=np.uint8)


@pytest.fixture
def rgb_image_640x480():
    """480-high, 640-wide (VGA) RGB image."""
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def square_box_points():
    """Simple axis-aligned 4x2 box: (10,10)-(110,10)-(110,60)-(10,60)."""
    return np.array([[10, 10], [110, 10], [110, 60], [10, 60]], dtype=np.float32)


@pytest.fixture
def simple_vocab():
    """Minimal CTC vocabulary: blank + a-z."""
    return ["blank"] + list("abcdefghijklmnopqrstuvwxyz")
