"""Tests for mlx_ppocr.processing.det_postprocess."""

import numpy as np

from mlx_ppocr.processing.det_postprocess import (
    _box_score,
    _order_points,
    _unclip,
    det_postprocess,
)


class TestOrderPoints:
    def test_already_ordered(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
        result = _order_points(pts)
        np.testing.assert_array_equal(result[0], [0, 0])   # TL
        np.testing.assert_array_equal(result[1], [10, 0])   # TR
        np.testing.assert_array_equal(result[2], [10, 10])  # BR
        np.testing.assert_array_equal(result[3], [0, 10])   # BL

    def test_shuffled_points(self):
        # Points given in random order
        pts = np.array([[10, 10], [0, 0], [0, 10], [10, 0]], dtype=np.int32)
        result = _order_points(pts)
        np.testing.assert_array_equal(result[0], [0, 0])
        np.testing.assert_array_equal(result[1], [10, 0])
        np.testing.assert_array_equal(result[2], [10, 10])
        np.testing.assert_array_equal(result[3], [0, 10])

    def test_non_square_rectangle(self):
        pts = np.array([[5, 0], [100, 0], [100, 20], [5, 20]], dtype=np.int32)
        result = _order_points(pts)
        np.testing.assert_array_equal(result[0], [5, 0])
        np.testing.assert_array_equal(result[1], [100, 0])


class TestBoxScore:
    def test_uniform_prob_map(self):
        prob = np.ones((50, 50), dtype=np.float32) * 0.8
        pts = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.int32)
        score = _box_score(prob, pts)
        assert abs(score - 0.8) < 1e-5

    def test_zero_prob_map(self):
        prob = np.zeros((50, 50), dtype=np.float32)
        pts = np.array([[5, 5], [20, 5], [20, 20], [5, 20]], dtype=np.int32)
        score = _box_score(prob, pts)
        assert score == 0.0

    def test_out_of_bounds_points_clipped(self):
        prob = np.ones((10, 10), dtype=np.float32) * 0.5
        pts = np.array([[-5, -5], [20, -5], [20, 20], [-5, 20]], dtype=np.int32)
        score = _box_score(prob, pts)
        # Should not crash, score computed on clipped region
        assert 0.0 <= score <= 1.0

    def test_degenerate_single_point(self):
        prob = np.ones((10, 10), dtype=np.float32)
        # All points collapse to one pixel — still a valid 1x1 region
        pts = np.array([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=np.int32)
        score = _box_score(prob, pts)
        assert 0.0 <= score <= 1.0


class TestUnclip:
    def test_basic_expansion(self):
        pts = np.array([[10, 10], [50, 10], [50, 40], [10, 40]], dtype=np.float32)
        result = _unclip(pts, 1.5)
        assert result is not None
        # Expanded polygon should be bigger
        from shapely.geometry import Polygon

        orig_area = Polygon(pts).area
        new_area = Polygon(result).area
        assert new_area > orig_area

    def test_zero_area_returns_none(self):
        # Collinear points → zero area
        pts = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32)
        assert _unclip(pts, 1.5) is None

    def test_small_polygon_returns_none(self):
        # Very tiny area < 1
        pts = np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]], dtype=np.float32)
        assert _unclip(pts, 1.5) is None


class TestDetPostprocess:
    def test_empty_prob_map(self):
        prob = np.zeros((64, 64), dtype=np.float32)
        result = det_postprocess(prob, 100, 100, 64, 64)
        assert result == []

    def test_single_blob(self):
        # Create a probability map with one bright square
        prob = np.zeros((64, 64), dtype=np.float32)
        prob[20:40, 20:40] = 0.9
        result = det_postprocess(
            prob, src_h=128, src_w=128,
            resize_h=64, resize_w=64,
            thresh=0.3, box_thresh=0.5,
        )
        assert len(result) >= 1
        for r in result:
            assert "points" in r
            assert "score" in r
            assert r["points"].shape == (4, 2)
            assert r["score"] >= 0.5

    def test_scaling_to_source_dimensions(self):
        prob = np.zeros((64, 64), dtype=np.float32)
        prob[10:50, 10:50] = 0.95
        result = det_postprocess(
            prob, src_h=640, src_w=640,
            resize_h=64, resize_w=64,
            thresh=0.3, box_thresh=0.5,
        )
        if result:
            pts = result[0]["points"]
            # Points should be in source image coordinates (up to 640)
            assert pts[:, 0].max() <= 640
            assert pts[:, 1].max() <= 640

    def test_box_thresh_filters_low_confidence(self):
        prob = np.zeros((64, 64), dtype=np.float32)
        prob[20:40, 20:40] = 0.4  # moderate probability
        result_high = det_postprocess(
            prob, 64, 64, 64, 64,
            thresh=0.3, box_thresh=0.8,
        )
        result_low = det_postprocess(
            prob, 64, 64, 64, 64,
            thresh=0.3, box_thresh=0.2,
        )
        assert len(result_high) <= len(result_low)

    def test_max_candidates_limits_output(self):
        prob = np.zeros((128, 128), dtype=np.float32)
        # Create many small blobs
        for y in range(0, 120, 15):
            for x in range(0, 120, 15):
                prob[y : y + 10, x : x + 10] = 0.9
        result = det_postprocess(
            prob, 128, 128, 128, 128,
            thresh=0.3, box_thresh=0.5, max_candidates=3,
        )
        assert len(result) <= 3

    def test_points_are_clipped_to_image(self):
        prob = np.zeros((64, 64), dtype=np.float32)
        prob[0:10, 0:10] = 0.95  # corner blob, expansion will go out of bounds
        result = det_postprocess(
            prob, 100, 100, 64, 64,
            thresh=0.3, box_thresh=0.5,
        )
        for r in result:
            assert r["points"][:, 0].min() >= 0
            assert r["points"][:, 1].min() >= 0
            assert r["points"][:, 0].max() <= 100
            assert r["points"][:, 1].max() <= 100
