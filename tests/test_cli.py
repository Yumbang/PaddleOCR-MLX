"""Tests for mlx_ppocr.cli (pure-logic functions)."""

import json
from argparse import Namespace


from mlx_ppocr.cli import (
    _build_parser,
    _error_json,
    _filter_fields,
    _format_json,
    _format_text,
)


class TestBuildParser:
    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["image.png"])
        assert args.images == ["image.png"]
        assert args.json_mode is False
        assert args.det_thresh == 0.1
        assert args.box_thresh == 0.3
        assert args.unclip_ratio == 1.5
        assert args.min_confidence == 0.0
        assert args.lang == "server"

    def test_json_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--json", "img.png"])
        assert args.json_mode is True

    def test_pdf_flags(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--force-ocr", "--dpi", "150", "--pages", "1-3", "doc.pdf",
        ])
        assert args.force_ocr is True
        assert args.dpi == 150
        assert args.pages == "1-3"

    def test_pdf_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["doc.pdf"])
        assert args.force_ocr is False
        assert args.dpi == 300
        assert args.pages is None

    def test_detection_params(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--det-thresh", "0.5",
            "--box-thresh", "0.7",
            "--unclip-ratio", "2.0",
            "--min-confidence", "0.8",
            "img.png",
        ])
        assert args.det_thresh == 0.5
        assert args.box_thresh == 0.7
        assert args.unclip_ratio == 2.0
        assert args.min_confidence == 0.8

    def test_multiple_images(self):
        parser = _build_parser()
        args = parser.parse_args(["a.png", "b.jpg", "c.pdf"])
        assert args.images == ["a.png", "b.jpg", "c.pdf"]

    def test_lang_option(self):
        parser = _build_parser()
        args = parser.parse_args(["--lang", "korean", "img.png"])
        assert args.lang == "korean"


class TestFilterFields:
    def test_filters_results(self):
        result = {
            "file": "img.png",
            "results": [
                {"text": "hello", "confidence": 0.9, "box": [[0, 0]]},
            ],
        }
        filtered = _filter_fields(result, ["text"])
        assert filtered["results"] == [{"text": "hello"}]

    def test_multiple_fields(self):
        result = {
            "results": [
                {"text": "hi", "confidence": 0.8, "box": [[1, 1]]},
            ],
        }
        filtered = _filter_fields(result, ["text", "confidence"])
        assert filtered["results"] == [{"text": "hi", "confidence": 0.8}]

    def test_no_results_key_unchanged(self):
        result = {"error": "something"}
        filtered = _filter_fields(result, ["text"])
        assert filtered == result

    def test_empty_results(self):
        result = {"results": []}
        filtered = _filter_fields(result, ["text"])
        assert filtered["results"] == []


class TestFormatJson:
    def test_compact(self):
        args = Namespace(pretty=False)
        result = {"text": "hello"}
        output = _format_json(result, args)
        assert output == '{"text": "hello"}'

    def test_pretty(self):
        args = Namespace(pretty=True)
        result = {"text": "hello"}
        output = _format_json(result, args)
        assert "\n" in output
        assert "  " in output

    def test_unicode_preserved(self):
        args = Namespace(pretty=False)
        result = {"text": "한글테스트"}
        output = _format_json(result, args)
        assert "한글테스트" in output
        assert "\\u" not in output


class TestFormatText:
    def test_error_result(self):
        result = {"file": "bad.png", "error": "File not found"}
        output = _format_text(result)
        assert "Error" in output
        assert "bad.png" in output
        assert "File not found" in output

    def test_image_result(self):
        result = {
            "file": "img.png",
            "image_size": {"width": 800, "height": 600},
            "processing_time_ms": 150,
            "result_count": 2,
            "results": [
                {"text": "hello", "confidence": 0.95},
                {"text": "world", "confidence": 0.88},
            ],
        }
        output = _format_text(result)
        assert "img.png" in output
        assert "800x600" in output
        assert "150ms" in output
        assert "2 text regions" in output
        assert "hello" in output
        assert "0.95" in output

    def test_pdf_result_shows_page_info(self):
        result = {
            "file": "doc.pdf",
            "page": 3,
            "page_count": 10,
            "page_size": {"width": 2550, "height": 3300},
            "processing_time_ms": 500,
            "result_count": 5,
            "embedded_count": 3,
            "ocr_count": 2,
            "results": [
                {"text": "embedded text", "confidence": 1.0, "source": "embedded"},
                {"text": "ocr text", "confidence": 0.9, "source": "ocr"},
            ],
        }
        output = _format_text(result)
        assert "Page: 3/10" in output
        assert "3 embedded" in output
        assert "2 OCR" in output
        assert "[embedded]" in output
        assert "[ocr]" in output

    def test_source_tag_only_when_present(self):
        result = {
            "file": "img.png",
            "image_size": {"width": 100, "height": 100},
            "processing_time_ms": 50,
            "result_count": 1,
            "results": [{"text": "no source", "confidence": 0.9}],
        }
        output = _format_text(result)
        assert "[embedded]" not in output
        assert "[ocr]" not in output


class TestErrorJson:
    def test_basic_error(self):
        args = Namespace(pretty=False)
        output = _error_json("Something failed", 1, args)
        data = json.loads(output)
        assert data["error"] == "Something failed"
        assert data["exit_code"] == 1

    def test_pretty_error(self):
        args = Namespace(pretty=True)
        output = _error_json("Error", 2, args)
        assert "\n" in output
