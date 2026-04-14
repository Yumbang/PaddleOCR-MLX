"""Agent-friendly CLI for mlx-ppocr.

Install globally:  uv tool install mlx-ppocr
Usage:             mlx-ocr --json image.png
"""

import argparse
import io
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

__version__ = "0.1.0"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INPUT = 2
EXIT_MODEL = 3
EXIT_PARTIAL = 4


@contextmanager
def _suppress_stdout():
    """Redirect stdout to devnull (suppresses MLXOCR print() during init)."""
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-ocr",
        description="PP-OCRv5 text recognition on Apple MLX. Extracts text from images.",
        epilog="""\
output formats:
  text    human-readable (default)
  json    one JSON object per image (JSONL for batch)

json schema:
  {"file": "img.png", "image_size": {"width": W, "height": H},
   "processing_time_ms": N, "result_count": N,
   "results": [{"text": "...", "confidence": 0.99,
                 "box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}]}

exit codes:
  0  success
  1  general error
  2  input error (file not found, invalid image)
  3  model error (weights missing, load failure)
  4  partial failure (some images failed in batch)

examples:
  mlx-ocr photo.png                             human-readable
  mlx-ocr --json photo.png                      structured JSON
  mlx-ocr --json --pretty photo.png             indented JSON
  mlx-ocr --json *.png                          batch JSONL
  mlx-ocr --json --fields text *.png            text only
  cat photo.png | mlx-ocr --json --stdin-image  pipe binary image
  find . -name '*.png' | mlx-ocr --json -       pipe file paths
  mlx-ocr --dry-run *.png                       validate inputs
  mlx-ocr --json doc.pdf                        PDF hybrid OCR
  mlx-ocr --json --force-ocr doc.pdf            PDF full OCR
  mlx-ocr --json --pages 1-3 doc.pdf            PDF page range
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "images", nargs="*", default=[],
        help='image paths, or "-" to read paths from stdin',
    )

    inp = parser.add_argument_group("input")
    inp.add_argument("--stdin-image", action="store_true", help="read binary image from stdin")
    inp.add_argument("--from-file", metavar="FILE", help="read image paths from file (one per line)")

    out = parser.add_argument_group("output")
    out.add_argument("--json", dest="json_mode", action="store_true", help="JSON output")
    out.add_argument("--output", choices=["text", "json"], default="text", help="output format")
    out.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    out.add_argument("--fields", help="comma-separated fields: text,confidence,box")
    out.add_argument("--quiet", action="store_true", help="suppress non-output messages")
    out.add_argument("--verbose", action="store_true", help="show detailed info")

    det = parser.add_argument_group("detection")
    det.add_argument("--det-thresh", type=float, default=0.1, help="binarization threshold")
    det.add_argument("--box-thresh", type=float, default=0.3, help="box confidence threshold")
    det.add_argument("--unclip-ratio", type=float, default=1.5, help="polygon expansion ratio")
    det.add_argument("--min-confidence", type=float, default=0.0, help="minimum OCR confidence")

    mdl = parser.add_argument_group("model")
    mdl.add_argument(
        "--lang", default="server",
        help="language/model preset (default: server). "
             "Options: server, mobile, korean, latin, cyrillic, arabic, "
             "devanagari, thai, greek, tamil, telugu, english, eslav. "
             "Aliases: japanese→mobile, chinese→server, spanish/french/"
             "german/italian/portuguese→latin, russian→cyrillic, "
             "hindi→devanagari, persian→arabic",
    )
    mdl.add_argument("--det-weights", help="detection weights path")
    mdl.add_argument("--rec-weights", help="recognition weights path")
    mdl.add_argument("--vocab", help="vocabulary file path")
    mdl.add_argument("--cache-dir", default="weights", help="weight cache directory")

    pdf = parser.add_argument_group("pdf")
    pdf.add_argument("--force-ocr", action="store_true", help="force full OCR on PDF pages, ignore embedded text")
    pdf.add_argument("--dpi", type=int, default=300, help="DPI for PDF rasterization (default: 300)")
    pdf.add_argument("--pages", metavar="STR", help="page range: '1-5' or '1,3,7' (default: all)")

    parser.add_argument("--visualize", action="store_true", help="save annotated image")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without running")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    return parser


def _resolve_images(args) -> list[tuple[str, str | Image.Image]]:
    """Resolve all image sources into (label, source) pairs."""
    images = []

    if args.stdin_image:
        data = sys.stdin.buffer.read()
        img = Image.open(io.BytesIO(data))
        images.append(("<stdin>", img))
        return images

    paths = list(args.images)

    if args.from_file:
        with open(args.from_file) as f:
            paths.extend(line.strip() for line in f if line.strip())

    for p in paths:
        if p == "-":
            for line in sys.stdin:
                line = line.strip()
                if line:
                    images.append((line, line))
        else:
            images.append((p, p))

    return images


def _process_one(ocr, label: str, source, args) -> dict:
    """Process a single image. Returns result dict."""
    if isinstance(source, Image.Image):
        img = source
    else:
        if not Path(source).is_file():
            return {"file": label, "error": f"File not found: {source}", "exit_code": EXIT_INPUT}
        try:
            img = Image.open(source).convert("RGB")
        except Exception as e:
            return {"file": label, "error": f"Invalid image: {e}", "exit_code": EXIT_INPUT}

    width, height = img.size

    t0 = time.time()
    results = ocr(
        img,
        det_thresh=args.det_thresh,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    entries = []
    for box, text, conf in results:
        if conf < args.min_confidence:
            continue
        entries.append({
            "text": text,
            "confidence": round(conf, 4),
            "box": box.tolist(),
        })

    return {
        "file": label,
        "image_size": {"width": width, "height": height},
        "processing_time_ms": elapsed_ms,
        "result_count": len(entries),
        "results": entries,
    }


def _filter_fields(result: dict, fields: list[str]) -> dict:
    """Keep only specified fields in each result entry."""
    if "results" in result:
        result = dict(result)
        result["results"] = [{k: v for k, v in r.items() if k in fields} for r in result["results"]]
    return result


def _format_json(result: dict, args) -> str:
    indent = 2 if args.pretty else None
    return json.dumps(result, ensure_ascii=False, indent=indent)


def _format_text(result: dict) -> str:
    if "error" in result:
        return f"Error [{result['file']}]: {result['error']}"
    lines = [f"File: {result['file']}"]
    if "page" in result:
        lines.append(f"Page: {result['page']}/{result['page_count']}")
        sz = result["page_size"]
    else:
        sz = result["image_size"]
    lines.append(f"Size: {sz['width']}x{sz['height']}  Time: {result['processing_time_ms']}ms")
    if "embedded_count" in result:
        lines.append(
            f"Found {result['result_count']} text regions "
            f"({result['embedded_count']} embedded, {result['ocr_count']} OCR):"
        )
    else:
        lines.append(f"Found {result['result_count']} text regions:")
    for i, r in enumerate(result["results"]):
        source_tag = f" [{r['source']}]" if "source" in r else ""
        lines.append(f"  [{i + 1}] ({r['confidence']:.2f}){source_tag} {r['text']}")
    return "\n".join(lines)


def _emit(result: dict, args):
    """Output a single result in the chosen format."""
    fields = args.fields.split(",") if args.fields else None
    if fields:
        result = _filter_fields(result, fields)

    use_json = args.json_mode or args.output == "json"
    if use_json:
        print(_format_json(result, args), flush=True)
    else:
        print(_format_text(result), flush=True)


def _error_json(message: str, code: int, args) -> str:
    d = {"error": message, "exit_code": code}
    indent = 2 if args.pretty else None
    return json.dumps(d, indent=indent)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    use_json = args.json_mode or args.output == "json"

    # Resolve images
    try:
        images = _resolve_images(args)
    except Exception as e:
        if use_json:
            print(_error_json(str(e), EXIT_INPUT, args))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(EXIT_INPUT)

    if not images:
        parser.print_help()
        sys.exit(EXIT_INPUT)

    # Dry run
    if args.dry_run:
        for label, source in images:
            if isinstance(source, str):
                exists = Path(source).is_file()
                status = "ok" if exists else "NOT FOUND"
            else:
                status = "ok (stdin)"
            print(f"  {status}: {label}")
        print(f"\n{len(images)} image(s) would be processed.")
        sys.exit(EXIT_OK)

    # Load model
    try:
        from mlx_ppocr.pipeline import MLXOCR

        suppress = use_json or args.quiet
        ocr_kwargs = dict(
            lang=args.lang,
            det_weights=args.det_weights,
            rec_weights=args.rec_weights,
            vocab_path=args.vocab,
            cache_dir=args.cache_dir,
        )
        if suppress:
            with _suppress_stdout():
                ocr = MLXOCR(**ocr_kwargs)
        else:
            ocr = MLXOCR(**ocr_kwargs)
    except Exception as e:
        if use_json:
            print(_error_json(f"Model load failed: {e}", EXIT_MODEL, args))
        else:
            print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(EXIT_MODEL)

    # Process images
    had_errors = False
    for label, source in images:
        try:
            if isinstance(source, str) and source.lower().endswith(".pdf"):
                from mlx_ppocr.pdf import process_pdf_hybrid

                page_results = process_pdf_hybrid(
                    ocr, source,
                    force_ocr=args.force_ocr,
                    dpi=args.dpi,
                    pages=args.pages,
                    det_thresh=args.det_thresh,
                    box_thresh=args.box_thresh,
                    unclip_ratio=args.unclip_ratio,
                    min_confidence=args.min_confidence,
                )
                for result in page_results:
                    if "error" in result:
                        had_errors = True
                    _emit(result, args)
                continue

            result = _process_one(ocr, label, source, args)
        except ImportError as e:
            result = {"file": label, "error": str(e), "exit_code": EXIT_ERROR}
        except Exception as e:
            result = {"file": label, "error": str(e), "exit_code": EXIT_ERROR}

        if "error" in result:
            had_errors = True

        _emit(result, args)

        if (args.visualize and "error" not in result
                and isinstance(source, str)
                and not source.lower().endswith(".pdf")):
            _visualize(source, result)

    if had_errors and len(images) > 1:
        sys.exit(EXIT_PARTIAL)
    elif had_errors:
        sys.exit(result.get("exit_code", EXIT_ERROR))


def _visualize(image_path: str, result: dict):
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    for r in result["results"]:
        pts = np.array(r["box"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    out_path = image_path.rsplit(".", 1)[0] + "_ocr.jpg"
    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    main()
