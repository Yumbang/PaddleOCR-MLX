"""Hybrid PDF OCR: extract embedded text where available, OCR the rest."""

import time

import numpy as np
from PIL import Image


def process_pdf_hybrid(
    ocr,
    pdf_path: str,
    *,
    force_ocr: bool = False,
    dpi: int = 300,
    pages: str | None = None,
    det_thresh: float = 0.1,
    box_thresh: float = 0.3,
    unclip_ratio: float = 1.5,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Process PDF with hybrid embedded+OCR approach. Returns per-page results."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PDF support requires PyMuPDF.\nInstall with:\n"
            "  uv pip install 'mlx-ppocr[pdf]'\n"
            "  # or\n"
            "  uv pip install pymupdf"
        ) from None

    doc = fitz.open(pdf_path)
    try:
        page_count = len(doc)
        page_indices = (
            _parse_page_range(pages, page_count) if pages else list(range(page_count))
        )

        results = []
        for page_idx in page_indices:
            page = doc[page_idx]
            result = _process_page(
                ocr, page, pdf_path, page_idx, page_count,
                force_ocr=force_ocr, dpi=dpi,
                det_thresh=det_thresh, box_thresh=box_thresh,
                unclip_ratio=unclip_ratio, min_confidence=min_confidence,
            )
            results.append(result)
    finally:
        doc.close()

    return results


def _process_page(
    ocr, page, pdf_path, page_idx, page_count, *,
    force_ocr, dpi, det_thresh, box_thresh, unclip_ratio, min_confidence,
) -> dict:
    """Single page: extract embedded -> rasterize -> detect -> coverage check -> OCR uncovered -> merge."""
    t0 = time.time()

    # Scale factor: PyMuPDF coordinates are in points (72 DPI)
    dpi_scale = dpi / 72.0

    # Step 1: Extract embedded text (unless force_ocr)
    embedded_lines = []
    if not force_ocr:
        words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
        embedded_lines = group_words_to_lines(words, dpi_scale)

    # Step 2: Rasterize page
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img_np = np.array(img)

    # Step 3: Run text detection
    det_boxes = ocr._detect(img_np, det_thresh, box_thresh, unclip_ratio)

    # Step 4: Coverage check + OCR uncovered regions
    from mlx_ppocr.processing.rec_preprocess import crop_text_region

    uncovered_crops = []
    uncovered_points = []
    for box_info in det_boxes:
        points = box_info["points"]
        w = max(np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[3] - points[2]))
        h = max(np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]))
        if w < 5 or h < 5:
            continue

        if not force_ocr and is_covered(points, embedded_lines):
            continue

        crop = crop_text_region(img_np, points)
        if crop is None or crop.shape[0] < 3 or crop.shape[1] < 3:
            continue

        uncovered_crops.append(crop)
        uncovered_points.append(points)

    # Step 5: Batch recognize uncovered regions
    ocr_results_list = []
    if uncovered_crops:
        rec_results = ocr._recognize_batch(uncovered_crops)
        for points, (text, score) in zip(uncovered_points, rec_results):
            if text and score > 0.5 and score >= min_confidence:
                ocr_results_list.append({
                    "text": text,
                    "confidence": round(score, 4),
                    "box": points.tolist(),
                    "source": "ocr",
                })

    # Step 6: Build embedded results (confidence is always 1.0 for embedded text)
    embedded_results = [
        {
            "text": line["text"],
            "confidence": 1.0,
            "box": line["box"],
            "source": "embedded",
        }
        for line in embedded_lines
    ]

    # Step 7: Merge and sort by position (y then x)
    all_results = embedded_results + ocr_results_list
    all_results.sort(key=lambda r: (min(p[1] for p in r["box"]), min(p[0] for p in r["box"])))

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "file": pdf_path,
        "page": page_idx + 1,
        "page_count": page_count,
        "page_size": {"width": pix.width, "height": pix.height},
        "processing_time_ms": elapsed_ms,
        "result_count": len(all_results),
        "embedded_count": len(embedded_results),
        "ocr_count": len(ocr_results_list),
        "results": all_results,
    }


def group_words_to_lines(words, dpi_scale: float) -> list[dict]:
    """Group PyMuPDF words by (block_no, line_no) into line-level results.

    Args:
        words: PyMuPDF word tuples (x0, y0, x1, y1, word, block_no, line_no, word_no)
        dpi_scale: Scale factor from 72 DPI to target DPI
    """
    lines: dict[tuple[int, int], list] = {}
    for w in words:
        x0, y0, x1, y1, text, block_no, line_no, _word_no = w
        key = (block_no, line_no)
        if key not in lines:
            lines[key] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "words": []}
        line = lines[key]
        line["x0"] = min(line["x0"], x0)
        line["y0"] = min(line["y0"], y0)
        line["x1"] = max(line["x1"], x1)
        line["y1"] = max(line["y1"], y1)
        line["words"].append((int(_word_no), text))

    result = []
    for line in lines.values():
        # Sort words by word_no and join
        line["words"].sort(key=lambda w: w[0])
        text = " ".join(w[1] for w in line["words"])

        # Scale coordinates from 72 DPI to target DPI
        x0 = line["x0"] * dpi_scale
        y0 = line["y0"] * dpi_scale
        x1 = line["x1"] * dpi_scale
        y1 = line["y1"] * dpi_scale

        box = [
            [int(x0), int(y0)],
            [int(x1), int(y0)],
            [int(x1), int(y1)],
            [int(x0), int(y1)],
        ]
        result.append({"text": text, "box": box})

    return result


def is_covered(
    det_box: np.ndarray,
    embedded_lines: list[dict],
    iou_thresh: float = 0.3,
    containment_thresh: float = 0.7,
) -> bool:
    """Check if detected region overlaps sufficiently with any embedded text line."""
    # Convert det_box (4x2 points) to axis-aligned bounds
    dx0 = det_box[:, 0].min()
    dy0 = det_box[:, 1].min()
    dx1 = det_box[:, 0].max()
    dy1 = det_box[:, 1].max()
    det_area = max((dx1 - dx0) * (dy1 - dy0), 1e-6)

    for line in embedded_lines:
        box = line["box"]
        ex0, ey0 = box[0]
        ex1, ey1 = box[2]

        # Intersection
        ix0 = max(dx0, ex0)
        iy0 = max(dy0, ey0)
        ix1 = min(dx1, ex1)
        iy1 = min(dy1, ey1)

        if ix0 >= ix1 or iy0 >= iy1:
            continue

        inter = (ix1 - ix0) * (iy1 - iy0)
        emb_area = max((ex1 - ex0) * (ey1 - ey0), 1e-6)
        union = det_area + emb_area - inter

        iou = inter / union
        if iou >= iou_thresh:
            return True

        # Containment: what fraction of det_box is covered by embedded text
        containment = inter / det_area
        if containment >= containment_thresh:
            return True

    return False


def _parse_page_range(pages_str: str, total: int) -> list[int]:
    """Parse '1-5' or '1,3,7' into 0-indexed page list.

    Raises:
        ValueError: If the page range string contains non-numeric content.
    """
    result = []
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = max(1, int(start_s.strip()))
            end = min(total, int(end_s.strip()))
            result.extend(range(start - 1, end))
        else:
            idx = int(part) - 1
            if 0 <= idx < total:
                result.append(idx)
    return sorted(set(result))
