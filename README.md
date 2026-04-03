# PaddleOCR-MLX

**PP-OCRv5 Server (best accuracy) ported to Apple's [MLX](https://github.com/ml-explore/mlx) framework for native Apple Silicon inference.**

No PyTorch. No PaddlePaddle. No ONNX. Just pure MLX — runs natively on M1/M2/M3/M4 with unified memory.

## Features

- **Full PP-OCRv5 Server pipeline** — detection + recognition, the highest-accuracy variant
- **Native Apple Silicon** — MLX backend, no framework translation layers
- **Auto weight download** — fetches and converts HuggingFace weights on first run
- **Batched recognition** — crops sorted by width and processed in batches (ported from PaddleOCR)
- **Agent-friendly CLI** — JSON output, stdin pipes, exit codes, self-documenting `--help`
- **~5 FPS** on Apple Silicon for 800×400 images

## Installation

```bash
# Install as a global CLI tool
uv tool install paddleocr-mlx

# Or install from source
git clone https://github.com/YOUR_USERNAME/PaddleOCR-MLX.git
cd PaddleOCR-MLX
uv tool install .
```

## Quick Start

### Python API

```python
from mlx_ppocr import MLXOCR

ocr = MLXOCR()
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
  --det-weights PATH     Detection weights path
  --rec-weights PATH     Recognition weights path
  --vocab PATH           Vocabulary file path
  --cache-dir DIR        Weight cache directory (default: weights)

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

## Architecture

The port faithfully reproduces the PP-OCRv5 Server architecture:

**Detection:**
PPHGNetV2-L backbone → LKPAN neck → PFHeadLocal (DB) head

**Recognition:**
PPHGNetV2-L backbone → AvgPool2d → SVTR encoder (transformer) → CTC head

Weights are automatically downloaded from HuggingFace ([det](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det_safetensors), [rec](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_safetensors)) and converted to MLX format on first run. Key conversion details:

- Conv2d weights: OIHW → OHWI (MLX layout)
- ConvTranspose2d weights: OIHW → IHWO
- All other parameters transferred directly

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
