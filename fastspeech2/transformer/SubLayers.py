"""Modified from ming024/FastSpeech2 ``transformer/SubLayers.py``.

Changes:
* The hand-written ``MultiHeadAttention`` / ``ScaledDotProductAttention`` were
  replaced by ``torch.nn.MultiheadAttention`` (batch-first, key-padding mask),
  which uses fused kernels and is used directly in ``Layers.FFTBlock``.
* ``PositionwiseFeedForward`` is unchanged in design (Conv1D kernels [9, 1],
  ReLU, dropout); the residual + LayerNorm moved into ``FFTBlock`` so the
  pre/post-norm order is visible in one place.
"""

from __future__ import annotations

import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):
    """Two 1D convolutions (kernel sizes [9, 1] by default) with ReLU in between."""

    def __init__(self, d_model: int, d_inner: int, kernel_sizes: tuple[int, int], dropout: float):
        super().__init__()
        k1, k2 = kernel_sizes
        self.conv1 = nn.Conv1d(d_model, d_inner, k1, padding=(k1 - 1) // 2)
        self.conv2 = nn.Conv1d(d_inner, d_model, k2, padding=(k2 - 1) // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(torch.relu(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        return self.dropout(y)
