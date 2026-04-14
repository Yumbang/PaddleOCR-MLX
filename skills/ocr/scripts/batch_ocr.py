#!/usr/bin/env python3
"""Batch OCR: recursively find images/PDFs in a directory, run mlx-ocr, aggregate results.

Outputs:
  - ocr_results.jsonl  : one JSON object per file (or per page for PDFs), structured
  - ocr_texts.txt      : plain extracted text grouped by file, for downstream consumption
  - summary printed to stdout

Usage:
  python batch_ocr.py <directory> [options]

Examples:
  python batch_ocr.py ./receipts
  python batch_ocr.py ./docs --lang server --ext pdf
  python batch_ocr.py ./scans --output-dir ./results --dpi 200
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS = {".pdf"}
ALL_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


def find_files(directory: str, extensions: set[str]) -> list[Path]:
    """Recursively find files matching the given extensions."""
    root = Path(directory)
    if not root.is_dir():
        print(f"Error: '{directory}' is not a directory", file=sys.stderr)
        sys.exit(1)

    files = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in extensions and path.is_file():
            files.append(path)
    return files


def run_ocr(file_path: Path, lang: str, dpi: int, force_ocr: bool) -> dict:
    """Run mlx-ocr on a single file and return parsed JSON results."""
    cmd = ["mlx-ocr", "--json", "--lang", lang, str(file_path)]
    if file_path.suffix.lower() in PDF_EXTENSIONS:
        cmd.extend(["--dpi", str(dpi)])
        if force_ocr:
            cmd.append("--force-ocr")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        # mlx-ocr outputs one JSON per line (JSONL for multi-page PDFs)
        outputs = []
        for line in result.stdout.strip().splitlines():
            if line.strip():
                try:
                    outputs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not outputs:
            return {
                "file": str(file_path),
                "error": result.stderr.strip() or "No output from mlx-ocr",
            }
        return outputs if len(outputs) > 1 else outputs[0]
    except subprocess.TimeoutExpired:
        return {"file": str(file_path), "error": "Timeout (300s)"}
    except FileNotFoundError:
        return {"file": str(file_path), "error": "mlx-ocr not found. Install with: uv pip install mlx-ppocr"}


def extract_text(result: dict) -> str:
    """Extract plain text from a single OCR result dict."""
    if "error" in result:
        return f"[ERROR: {result['error']}]"
    texts = []
    for r in result.get("results", []):
        texts.append(r["text"])
    return "\n".join(texts)


def main():
    parser = argparse.ArgumentParser(
        description="Batch OCR: recursively process images and PDFs with mlx-ocr",
    )
    parser.add_argument("directory", help="Directory to scan recursively")
    parser.add_argument("--lang", default="korean", help="OCR language (default: korean)")
    parser.add_argument("--dpi", type=int, default=300, help="PDF rasterization DPI (default: 300)")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR on PDFs (ignore embedded text)")
    parser.add_argument("--ext", choices=["all", "image", "pdf"], default="all", help="File types to process")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: current directory)")
    args = parser.parse_args()

    extensions = ALL_EXTENSIONS
    if args.ext == "image":
        extensions = IMAGE_EXTENSIONS
    elif args.ext == "pdf":
        extensions = PDF_EXTENSIONS

    files = find_files(args.directory, extensions)
    if not files:
        print(f"No matching files found in '{args.directory}'")
        sys.exit(0)

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "ocr_results.jsonl"
    text_path = output_dir / "ocr_texts.txt"

    print(f"Found {len(files)} file(s) in '{args.directory}'")
    print(f"Language: {args.lang} | DPI: {args.dpi} | Types: {args.ext}")
    print(f"Output: {jsonl_path}")
    print()

    success = 0
    errors = 0
    total_regions = 0
    t_start = time.time()

    with open(jsonl_path, "w") as f_jsonl, open(text_path, "w") as f_text:
        for i, fpath in enumerate(files, 1):
            rel = fpath.relative_to(Path(args.directory))
            print(f"  [{i}/{len(files)}] {rel} ... ", end="", flush=True)

            result = run_ocr(fpath, args.lang, args.dpi, args.force_ocr)

            # Handle multi-page PDFs (list of dicts)
            if isinstance(result, list):
                page_regions = 0
                has_error = False
                for page_result in result:
                    f_jsonl.write(json.dumps(page_result, ensure_ascii=False) + "\n")
                    if "error" in page_result:
                        has_error = True
                    else:
                        page_regions += page_result.get("result_count", 0)

                f_text.write(f"\n{'='*60}\n")
                f_text.write(f"FILE: {fpath}\n")
                f_text.write(f"{'='*60}\n")
                for page_result in result:
                    if "page" in page_result:
                        f_text.write(f"\n--- Page {page_result['page']} ---\n")
                    f_text.write(extract_text(page_result) + "\n")

                if has_error:
                    errors += 1
                    print("ERROR")
                else:
                    success += 1
                    total_regions += page_regions
                    page_count = len(result)
                    print(f"{page_count} pages, {page_regions} regions")
            else:
                f_jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")

                f_text.write(f"\n{'='*60}\n")
                f_text.write(f"FILE: {fpath}\n")
                f_text.write(f"{'='*60}\n")
                f_text.write(extract_text(result) + "\n")

                if "error" in result:
                    errors += 1
                    print(f"ERROR: {result['error']}")
                else:
                    n = result.get("result_count", 0)
                    total_regions += n
                    success += 1
                    print(f"{n} regions")

    elapsed = time.time() - t_start

    print()
    print("=" * 40)
    print(f"  Files processed: {success + errors}")
    print(f"  Successful:      {success}")
    print(f"  Errors:          {errors}")
    print(f"  Total regions:   {total_regions}")
    print(f"  Time:            {elapsed:.1f}s")
    print(f"  Results:         {jsonl_path}")
    print(f"  Plain text:      {text_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
