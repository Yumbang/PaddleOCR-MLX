"""Tests for mlx_ppocr.pdf (pure-logic functions)."""

import numpy as np
import pytest

from mlx_ppocr.pdf import _parse_page_range, group_words_to_lines, is_covered


class TestParsePageRange:
    def test_single_page(self):
        assert _parse_page_range("1", 10) == [0]

    def test_comma_separated(self):
        assert _parse_page_range("1,3,5", 10) == [0, 2, 4]

    def test_range(self):
        assert _parse_page_range("2-5", 10) == [1, 2, 3, 4]

    def test_mixed_range_and_single(self):
        result = _parse_page_range("1,3-5,8", 10)
        assert result == [0, 2, 3, 4, 7]

    def test_out_of_bounds_clamped(self):
        result = _parse_page_range("1-20", 5)
        assert result == [0, 1, 2, 3, 4]

    def test_page_beyond_total_ignored(self):
        result = _parse_page_range("99", 5)
        assert result == []

    def test_page_zero_ignored(self):
        result = _parse_page_range("0", 5)
        assert result == []

    def test_duplicates_removed(self):
        result = _parse_page_range("1,1,2,2", 5)
        assert result == [0, 1]

    def test_result_is_sorted(self):
        result = _parse_page_range("5,3,1", 10)
        assert result == [0, 2, 4]

    def test_spaces_handled(self):
        result = _parse_page_range(" 1 , 3 - 5 ", 10)
        assert result == [0, 2, 3, 4]

    def test_single_page_range(self):
        result = _parse_page_range("3-3", 10)
        assert result == [2]

    def test_reversed_range_returns_empty(self):
        result = _parse_page_range("5-2", 10)
        assert result == []

    def test_non_numeric_raises_value_error(self):
        with pytest.raises(ValueError):
            _parse_page_range("abc", 10)

    def test_empty_parts_skipped(self):
        result = _parse_page_range("1,,3", 10)
        assert result == [0, 2]


class TestGroupWordsToLines:
    def test_single_word(self):
        words = [
            (10.0, 20.0, 50.0, 35.0, "hello", 0, 0, 0),
        ]
        result = group_words_to_lines(words, dpi_scale=1.0)
        assert len(result) == 1
        assert result[0]["text"] == "hello"

    def test_words_on_same_line_joined(self):
        words = [
            (10.0, 20.0, 40.0, 35.0, "hello", 0, 0, 0),
            (45.0, 20.0, 80.0, 35.0, "world", 0, 0, 1),
        ]
        result = group_words_to_lines(words, dpi_scale=1.0)
        assert len(result) == 1
        assert result[0]["text"] == "hello world"

    def test_words_on_different_lines(self):
        words = [
            (10.0, 20.0, 50.0, 35.0, "line1", 0, 0, 0),
            (10.0, 40.0, 50.0, 55.0, "line2", 0, 1, 0),
        ]
        result = group_words_to_lines(words, dpi_scale=1.0)
        assert len(result) == 2
        texts = {r["text"] for r in result}
        assert texts == {"line1", "line2"}

    def test_different_blocks_are_separate(self):
        words = [
            (10.0, 20.0, 50.0, 35.0, "block0", 0, 0, 0),
            (10.0, 20.0, 50.0, 35.0, "block1", 1, 0, 0),
        ]
        result = group_words_to_lines(words, dpi_scale=1.0)
        assert len(result) == 2

    def test_dpi_scaling(self):
        words = [
            (10.0, 20.0, 50.0, 35.0, "scaled", 0, 0, 0),
        ]
        scale = 300.0 / 72.0
        result = group_words_to_lines(words, dpi_scale=scale)
        box = result[0]["box"]
        # box[0] is top-left [x, y]
        assert box[0][0] == int(10.0 * scale)
        assert box[0][1] == int(20.0 * scale)

    def test_bounding_box_merges_across_words(self):
        words = [
            (10.0, 20.0, 40.0, 35.0, "A", 0, 0, 0),
            (50.0, 18.0, 90.0, 37.0, "B", 0, 0, 1),
        ]
        result = group_words_to_lines(words, dpi_scale=1.0)
        box = result[0]["box"]
        # TL should be min of x0s, min of y0s
        assert box[0] == [10, 18]
        # BR should be max of x1s, max of y1s
        assert box[2] == [90, 37]

    def test_word_order_by_word_no(self):
        words = [
            (50.0, 20.0, 90.0, 35.0, "second", 0, 0, 1),
            (10.0, 20.0, 40.0, 35.0, "first", 0, 0, 0),
        ]
        result = group_words_to_lines(words, dpi_scale=1.0)
        assert result[0]["text"] == "first second"

    def test_empty_input(self):
        assert group_words_to_lines([], dpi_scale=1.0) == []

    def test_box_format_is_four_point(self):
        words = [(10.0, 20.0, 50.0, 35.0, "test", 0, 0, 0)]
        result = group_words_to_lines(words, dpi_scale=1.0)
        box = result[0]["box"]
        assert len(box) == 4
        assert all(len(pt) == 2 for pt in box)
        # Verify it's TL, TR, BR, BL
        assert box[0][0] < box[1][0]  # TL.x < TR.x
        assert box[0][1] < box[3][1]  # TL.y < BL.y


class TestIsCovered:
    def test_perfect_overlap(self):
        det_box = np.array([[10, 10], [100, 10], [100, 50], [10, 50]], dtype=np.float32)
        lines = [{"box": [[10, 10], [100, 10], [100, 50], [10, 50]]}]
        assert is_covered(det_box, lines) is True

    def test_no_overlap(self):
        det_box = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
        lines = [{"box": [[200, 200], [300, 200], [300, 250], [200, 250]]}]
        assert is_covered(det_box, lines) is False

    def test_partial_overlap_below_thresholds(self):
        det_box = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        # Small overlap region in corner
        lines = [{"box": [[90, 90], [120, 90], [120, 120], [90, 120]]}]
        assert is_covered(det_box, lines) is False

    def test_iou_threshold(self):
        # Two boxes with ~50% overlap → IoU > 0.3
        det_box = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        lines = [{"box": [[50, 0], [150, 0], [150, 50], [50, 50]]}]
        assert is_covered(det_box, lines, iou_thresh=0.3) is True

    def test_containment_threshold(self):
        # Detection box fully inside a larger embedded line
        det_box = np.array([[20, 20], [80, 20], [80, 40], [20, 40]], dtype=np.float32)
        lines = [{"box": [[0, 0], [100, 0], [100, 60], [0, 60]]}]
        assert is_covered(det_box, lines) is True

    def test_empty_embedded_lines(self):
        det_box = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
        assert is_covered(det_box, []) is False

    def test_custom_thresholds(self):
        det_box = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        lines = [{"box": [[0, 0], [100, 0], [100, 50], [0, 50]]}]
        # Even with very high thresholds, perfect overlap passes
        assert is_covered(det_box, lines, iou_thresh=0.99, containment_thresh=0.99) is True

    def test_multiple_lines_any_match(self):
        det_box = np.array([[50, 50], [100, 50], [100, 80], [50, 80]], dtype=np.float32)
        lines = [
            {"box": [[0, 0], [10, 0], [10, 10], [0, 10]]},       # no overlap
            {"box": [[50, 50], [100, 50], [100, 80], [50, 80]]},  # perfect match
        ]
        assert is_covered(det_box, lines) is True
