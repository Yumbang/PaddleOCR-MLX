---
name: ocr
description: >-
  Batch OCR processing of images and PDFs across directories using the local
  mlx-ocr CLI on Apple Silicon. Use this skill when the user wants to extract
  text from multiple files at once — scanning a folder, processing all receipts
  in a directory, OCR-ing nested document trees, or any task involving more
  than a handful of images/PDFs. This skill keeps bulk OCR output out of the
  conversation by writing results to disk and returning only a summary.
  Do NOT use this for a single image or PDF — Claude's built-in vision handles
  those better. Trigger on phrases like "OCR this folder", "extract text from
  all files in", "batch process these documents", "scan all PDFs in", or any
  request involving recursive/directory-level text extraction.
context: fork
allowed-tools: Bash(python *) Bash(mlx-ocr *) Bash(find *) Bash(wc *) Bash(cat *) Bash(head *) Bash(ls *) Read Glob Grep
---

# Batch OCR Skill

> **Note for repo visitors**: This skill is an example of how to use the
> `mlx-ocr` CLI (from the `mlx-ppocr` package) as a Claude Code skill for
> batch document processing. The default language is set to `korean` to match
> the repo author's primary use case — change `--lang` to suit your needs.
> Available languages: `server`, `mobile`, `korean`, `latin`, `cyrillic`,
> `arabic`, `devanagari`, `thai`, `english`, and more. Run `mlx-ocr --help`
> for the full list.

## When to use this skill

Use this skill for **bulk extraction** — directories of receipts, nested
folders of scanned documents, batch invoice processing. The key advantage
over Claude's built-in vision is throughput and cost: this runs locally on
Apple Silicon with zero API calls, and results are written to files rather
than consuming conversation context.

For a single image or PDF, do NOT use this skill. Just read the file directly
with Claude's vision — it gives richer understanding and can reason about
layout and context.

## How to use

Run the bundled batch script. It recursively finds images and PDFs, runs
`mlx-ocr` on each, and produces two output files:

```bash
python ${CLAUDE_SKILL_DIR}/scripts/batch_ocr.py <directory> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--lang` | `korean` | OCR language/model preset |
| `--dpi` | `300` | PDF rasterization DPI |
| `--force-ocr` | off | Force full OCR on PDFs (ignore embedded text) |
| `--ext` | `all` | Filter: `all`, `image`, or `pdf` |
| `--output-dir` | current dir | Where to write result files |

### Output files

- **`ocr_results.jsonl`** — One JSON object per file (or per page for PDFs).
  Contains bounding boxes, confidence scores, and source (embedded vs OCR).
  Use this for structured downstream processing.

- **`ocr_texts.txt`** — Plain extracted text grouped by file, separated by
  headers. Use this for search, summarization, or feeding into other tools.

### Examples

```bash
# OCR all images and PDFs in a receipts folder
python ${CLAUDE_SKILL_DIR}/scripts/batch_ocr.py ./receipts

# Process only PDFs with English model
python ${CLAUDE_SKILL_DIR}/scripts/batch_ocr.py ./documents --ext pdf --lang server

# Save results to a specific directory
python ${CLAUDE_SKILL_DIR}/scripts/batch_ocr.py ./scans --output-dir ./results

# Force OCR on PDFs (ignore embedded text)
python ${CLAUDE_SKILL_DIR}/scripts/batch_ocr.py ./mixed --force-ocr
```

## After processing

Once the batch script finishes, report the summary to the user (file count,
successes, errors, total text regions, elapsed time). Then offer to help with
the results:

- "Want me to search the extracted text for something specific?"
- "Should I summarize what's in these documents?"
- "Want me to extract structured data (dates, amounts, names) from the results?"

If the user wants to dig into specific files, read from `ocr_texts.txt` or
`ocr_results.jsonl` as needed — but only load what's relevant, not the
entire file.

## Troubleshooting

If `mlx-ocr` is not found, the user needs to install the package:

```bash
uv pip install mlx-ppocr
# For PDF support:
uv pip install 'mlx-ppocr[pdf]'
```

If OCR quality is poor, suggest:
- Try a different `--lang` preset (e.g., `server` for Chinese/English, `latin` for European languages)
- Increase `--dpi` for PDFs (e.g., `--dpi 600`)
- Use `--force-ocr` if a PDF has bad embedded text
