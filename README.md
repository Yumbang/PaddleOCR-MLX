# PaddleOCR-MLX

**PP-OCRv5 ported to Apple's [MLX](https://github.com/ml-explore/mlx) framework for native Apple Silicon inference. Supports 20+ languages.**

No PyTorch. No PaddlePaddle. No ONNX. Just pure MLX — runs natively on M1/M2/M3/M4 with unified memory.

## Features

- **Multi-language OCR** — server (Chinese/English), mobile (+ Japanese), Korean, Latin, Cyrillic, Arabic, and more
- **Two model variants** — server (best accuracy) and mobile (lightweight, multilingual)
- **Native Apple Silicon** — MLX backend, no framework translation layers
- **Auto weight download** — fetches and converts HuggingFace weights on first run
- **Batched recognition** — crops sorted by width and processed in batches (ported from PaddleOCR)
- **Hybrid PDF OCR** — extracts embedded text where available, runs OCR only on uncovered regions
- **Agent-friendly CLI** — JSON output, stdin pipes, exit codes, self-documenting `--help`
- **Claude Code skill** — batch OCR skill for processing entire directories as an AI agent tool
- **~5 FPS** on M3 Pro for 800×400 images

## Installation

```bash
# Install as a global CLI tool
uv tool install paddleocr-mlx

# Or install from source
git clone https://github.com/Yumbang/PaddleOCR-MLX.git
cd PaddleOCR-MLX
uv tool install .
```

## Quick Start

### Python API

```python
from mlx_ppocr import MLXOCR

ocr = MLXOCR()                        # server (Chinese + English)
ocr = MLXOCR(lang="mobile")           # mobile (Chinese + English + Japanese)
ocr = MLXOCR(lang="korean")           # Korean + English
ocr = MLXOCR(lang="latin")            # French, Spanish, German, etc.

results = ocr("photo.png")
for box, text, confidence in results:
    print(f"{text} ({confidence:.2f})")
```

### CLI

```bash
# Human-readable output
mlx-ocr photo.png

# Structured JSON
mlx-ocr --json photo.png

# Pretty-printed JSON
mlx-ocr --json --pretty photo.png

# Batch processing (JSONL)
mlx-ocr --json *.png

# Text-only output
mlx-ocr --json --fields text *.png

# Pipe binary image from stdin
cat photo.png | mlx-ocr --json --stdin-image

# Pipe file paths
find . -name '*.png' | mlx-ocr --json -

# Validate inputs without running OCR
mlx-ocr --dry-run *.png

# High-confidence results only
mlx-ocr --json --min-confidence 0.9 document.png

# Multi-language support
mlx-ocr --lang mobile photo.png       # Chinese + English + Japanese
mlx-ocr --lang korean receipt.png      # Korean + English
mlx-ocr --lang latin document.png      # French, Spanish, German, etc.
mlx-ocr --lang arabic sign.png         # Arabic, Persian, Urdu
```

### PDF OCR

```bash
# Hybrid OCR (uses embedded text where available, OCR for the rest)
mlx-ocr --json doc.pdf

# Force full OCR (ignore embedded text)
mlx-ocr --json --force-ocr doc.pdf

# Select specific pages
mlx-ocr --json --pages 1-3 doc.pdf

# Custom DPI for rasterization
mlx-ocr --json --dpi 600 doc.pdf

# Korean receipt PDF
mlx-ocr --json --pretty --lang korean receipt.pdf
```

PDF support requires PyMuPDF:

```bash
uv pip install 'mlx-ppocr[pdf]'
```

### JSON Output Schema

```json
{
  "file": "photo.png",
  "image_size": {"width": 800, "height": 400},
  "processing_time_ms": 200,
  "result_count": 2,
  "results": [
    {
      "text": "Hello World!",
      "confidence": 0.99,
      "box": [[95, 93], [430, 93], [430, 155], [95, 155]]
    }
  ]
}
```

For PDFs, each page is a separate JSON object with additional fields:

```json
{
  "file": "doc.pdf",
  "page": 1,
  "page_count": 10,
  "page_size": {"width": 2550, "height": 3300},
  "processing_time_ms": 450,
  "result_count": 15,
  "embedded_count": 12,
  "ocr_count": 3,
  "results": [
    {"text": "Invoice #12345", "confidence": 1.0, "box": [[...]], "source": "embedded"},
    {"text": "signature",      "confidence": 0.87, "box": [[...]], "source": "ocr"}
  ]
}
```

### CLI Options

```
mlx-ocr [OPTIONS] [IMAGES...]

input:
  --stdin-image          Read binary image from stdin
  --from-file FILE       Read image paths from file (one per line)

output:
  --json                 JSON output (JSONL for batch)
  --pretty               Pretty-print JSON
  --fields FIELDS        Comma-separated: text,confidence,box
  --quiet                Suppress non-output messages
  --verbose              Show detailed info

detection:
  --det-thresh FLOAT     Binarization threshold (default: 0.1)
  --box-thresh FLOAT     Box confidence threshold (default: 0.3)
  --unclip-ratio FLOAT   Polygon expansion ratio (default: 1.5)
  --min-confidence FLOAT Minimum OCR confidence (default: 0.0)

model:
  --lang LANG            Language/model preset (default: server)
  --det-weights PATH     Detection weights path
  --rec-weights PATH     Recognition weights path
  --vocab PATH           Vocabulary file path
  --cache-dir DIR        Weight cache directory (default: weights)

pdf:
  --force-ocr            Force full OCR on PDF (ignore embedded text)
  --dpi INT              PDF rasterization DPI (default: 300)
  --pages STR            Page range: '1-5' or '1,3,7' (default: all)

other:
  --visualize            Save annotated image
  --dry-run              Validate inputs without running
  --version              Print version
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Input error (file not found, invalid image) |
| 3 | Model error (weights missing, load failure) |
| 4 | Partial failure (some images failed in batch) |

## Supported Languages

| Preset | Languages | Model |
|--------|-----------|-------|
| `server` (default) | Chinese, English | Server (best accuracy) |
| `mobile` | Chinese, English, Japanese | Mobile |
| `korean` | Korean, English | Mobile |
| `latin` | French, Spanish, German, Italian, Portuguese, 40+ more | Mobile |
| `cyrillic` | Russian, Bulgarian, Ukrainian, 30+ more | Mobile |
| `arabic` | Arabic, Persian, Urdu, Kurdish | Mobile |
| `devanagari` | Hindi, Marathi, Nepali, Sanskrit | Mobile |
| `thai` | Thai, English | Mobile |
| `greek` | Greek, English | Mobile |
| `tamil` | Tamil, English | Mobile |
| `telugu` | Telugu, English | Mobile |

Aliases: `japanese`→mobile, `chinese`→server, `spanish`/`french`/`german`/`italian`/`portuguese`→latin, `russian`→cyrillic, `hindi`→devanagari, `persian`→arabic

The `server` and `mobile` presets use HuggingFace safetensors (zero extra dependencies). Other languages require a one-time weight conversion via `paddlepaddle`:

```bash
uv tool install mlx-ppocr[multilingual]   # includes paddlepaddle
```

## Claude Code Skill: Batch OCR

This repo includes an example [Claude Code skill](https://code.claude.com/docs/en/skills) that turns `mlx-ocr` into a batch processing tool for AI agents. The skill recursively scans directories, processes all images and PDFs, and writes results to disk — keeping bulk OCR output out of the conversation context.

### Why a skill instead of just calling the CLI?

For a **single image or PDF**, Claude's built-in vision is better — it understands layout, can reason about content, and needs no setup. The skill is for **bulk processing**: hundreds of receipts, nested folders of scanned documents, or batch invoice extraction. It runs locally on Apple Silicon with zero API cost per file.

### Install the skill

```bash
# Option 1: Use as a Claude Code plugin (recommended)
git clone https://github.com/Yumbang/PaddleOCR-MLX.git
claude --plugin-dir ./PaddleOCR-MLX    # test it out

# Option 2: Copy into your project (project-specific, no namespace)
cp -r PaddleOCR-MLX/skills/ocr .claude/skills/ocr
```

### Usage

```
# As a plugin (namespaced)
/mlx-ocr:ocr ~/Documents/receipts/2026/

# As a project skill (if copied to .claude/skills/)
/ocr ~/Documents/receipts/2026/

# Or just describe what you want — Claude auto-triggers on directory-level OCR
"Extract text from all PDFs in the expenses folder"
"OCR everything in ./scans/ and summarize what you find"
```

### What the skill does

1. Recursively finds images (`.png`, `.jpg`, `.tiff`, etc.) and PDFs in the target directory
2. Runs `mlx-ocr --json` on each file (default: `--lang korean`)
3. Writes structured results to `ocr_results.jsonl` and plain text to `ocr_texts.txt`
4. Returns a concise summary (file count, regions found, errors) to the conversation

The skill runs in a forked subagent (`context: fork`), so the OCR output never enters your main conversation context.

### Customizing

The skill defaults to `--lang korean`. To change the default language for your use case, edit the `--lang` flag in `.claude/skills/ocr/scripts/batch_ocr.py`:

```python
parser.add_argument("--lang", default="korean", ...)  # change to "server", "latin", etc.
```

## Architecture

**Detection** (shared across all languages):
PPHGNetV2-L backbone → LKPAN neck → PFHeadLocal (DB) head

**Recognition (server):**
PPHGNetV2-L backbone → AvgPool2d → SVTR encoder → CTC head

**Recognition (mobile):**
PP-LCNetV3 backbone → AvgPool2d → SVTR encoder → CTC head

Weights are automatically downloaded from HuggingFace and converted to MLX format on first run.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python ≥ 3.10

## Acknowledgments

- [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — the original PP-OCRv5 implementation
- [Apple MLX](https://github.com/ml-explore/mlx) — the ML framework for Apple Silicon
- [HuggingFace](https://huggingface.co/PaddlePaddle) — model weight hosting

## Built With

This project was built through a collaboration between a human developer and [Claude Code](https://claude.ai/claude-code) (Claude Opus 4.6). Claude ported the full model architecture layer-by-layer from PyTorch/PaddlePaddle to MLX, debugged weight conversion edge cases, implemented batched recognition, and designed the agent-friendly CLI.

## License

MIT
