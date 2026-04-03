"""CTC head for PP-OCRv5 server recognition."""

import mlx.nn as nn


class CTCHead(nn.Linear):
    """Simple linear projection: hidden_size → vocab_size."""

    def __init__(self, in_channels: int = 120, out_channels: int = 18385):
        super().__init__(in_channels, out_channels)
