"""Tests for mlx_ppocr.processing.rec_postprocess."""

import json

import numpy as np
import pytest

from mlx_ppocr.processing.rec_postprocess import _softmax, ctc_decode, load_vocab


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = _softmax(x)
        np.testing.assert_almost_equal(result.sum(axis=-1), 1.0)

    def test_all_zeros(self):
        x = np.array([[0.0, 0.0, 0.0]])
        result = _softmax(x)
        np.testing.assert_almost_equal(result, [[1 / 3, 1 / 3, 1 / 3]])

    def test_numerical_stability_large_values(self):
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = _softmax(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_almost_equal(result.sum(axis=-1), 1.0)

    def test_negative_values(self):
        x = np.array([[-100.0, -99.0, -98.0]])
        result = _softmax(x)
        np.testing.assert_almost_equal(result.sum(axis=-1), 1.0)
        # Largest value should have highest probability
        assert result[0, 2] > result[0, 1] > result[0, 0]

    def test_batch_dimension(self):
        x = np.random.randn(4, 10, 5)
        result = _softmax(x)
        assert result.shape == x.shape
        np.testing.assert_almost_equal(result.sum(axis=-1), np.ones((4, 10)))


class TestCtcDecode:
    def test_basic_decode(self, simple_vocab):
        # Create logits for "abc": blank, a, a, blank, b, blank, c
        B, T, V = 1, 7, len(simple_vocab)
        logits = np.full((B, T, V), -10.0)
        # Make each timestep strongly predict one character
        logits[0, 0, 0] = 10.0   # blank
        logits[0, 1, 1] = 10.0   # 'a'
        logits[0, 2, 1] = 10.0   # 'a' (duplicate → collapsed)
        logits[0, 3, 0] = 10.0   # blank
        logits[0, 4, 2] = 10.0   # 'b'
        logits[0, 5, 0] = 10.0   # blank
        logits[0, 6, 3] = 10.0   # 'c'
        results = ctc_decode(logits, simple_vocab)
        assert len(results) == 1
        text, conf = results[0]
        assert text == "abc"
        assert conf > 0.9

    def test_all_blank(self, simple_vocab):
        B, T, V = 1, 10, len(simple_vocab)
        logits = np.full((B, T, V), -10.0)
        logits[:, :, 0] = 10.0  # all blank
        results = ctc_decode(logits, simple_vocab)
        assert results[0][0] == ""
        assert results[0][1] == 0.0

    def test_duplicate_suppression(self, simple_vocab):
        # 'a' repeated 5 times without blanks → single 'a'
        B, T, V = 1, 5, len(simple_vocab)
        logits = np.full((B, T, V), -10.0)
        logits[0, :, 1] = 10.0  # all 'a'
        results = ctc_decode(logits, simple_vocab)
        assert results[0][0] == "a"

    def test_repeated_char_with_blank_separator(self, simple_vocab):
        # a, blank, a → "aa"
        B, T, V = 1, 3, len(simple_vocab)
        logits = np.full((B, T, V), -10.0)
        logits[0, 0, 1] = 10.0  # 'a'
        logits[0, 1, 0] = 10.0  # blank
        logits[0, 2, 1] = 10.0  # 'a'
        results = ctc_decode(logits, simple_vocab)
        assert results[0][0] == "aa"

    def test_batch_decode(self, simple_vocab):
        B, T, V = 3, 5, len(simple_vocab)
        logits = np.full((B, T, V), -10.0)
        # Batch 0: "a"
        logits[0, 0, 1] = 10.0
        logits[0, 1:, 0] = 10.0
        # Batch 1: "b"
        logits[1, 0, 2] = 10.0
        logits[1, 1:, 0] = 10.0
        # Batch 2: all blank
        logits[2, :, 0] = 10.0

        results = ctc_decode(logits, simple_vocab)
        assert len(results) == 3
        assert results[0][0] == "a"
        assert results[1][0] == "b"
        assert results[2][0] == ""

    def test_out_of_vocab_index_skipped(self):
        vocab = ["blank", "a", "b"]
        B, T, V = 1, 3, 10  # V > len(vocab)
        logits = np.full((B, T, V), -10.0)
        logits[0, 0, 1] = 10.0  # 'a' (valid)
        logits[0, 1, 8] = 10.0  # index 8 > len(vocab), skipped
        logits[0, 2, 2] = 10.0  # 'b' (valid)
        results = ctc_decode(logits, vocab)
        assert results[0][0] == "ab"


class TestLoadVocab:
    def test_hf_character_list_format(self, tmp_path):
        data = {"character_list": ["a", "b", "c"]}
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(data))
        result = load_vocab(str(path))
        assert result == ["a", "b", "c"]

    def test_hf_character_dict_format(self, tmp_path):
        data = {"character_dict": ["x", "y", "z"]}
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(data))
        result = load_vocab(str(path))
        assert result == ["x", "y", "z"]

    def test_paddleocr_format_prepends_blank(self, tmp_path):
        data = {"PostProcess": {"character_dict": ["가", "나", "다"]}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data))
        result = load_vocab(str(path))
        assert result[0] == "blank"
        assert result[1:] == ["가", "나", "다"]

    def test_text_file_format(self, tmp_path):
        path = tmp_path / "vocab.txt"
        path.write_text("a\nb\nc\n")
        result = load_vocab(str(path))
        # strip("\n") preserves content but removes trailing newline
        assert result == ["a", "b", "c"]

    def test_unknown_json_format_raises(self, tmp_path):
        data = {"unrelated_key": [1, 2, 3]}
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="No character list found"):
            load_vocab(str(path))

    def test_unicode_characters(self, tmp_path):
        data = {"character_list": ["你", "好", "世", "界"]}
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(data, ensure_ascii=False))
        result = load_vocab(str(path))
        assert result == ["你", "好", "世", "界"]

    def test_priority_character_list_over_dict(self, tmp_path):
        # character_list should take priority
        data = {
            "character_list": ["a", "b"],
            "character_dict": ["x", "y"],
        }
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(data))
        result = load_vocab(str(path))
        assert result == ["a", "b"]
