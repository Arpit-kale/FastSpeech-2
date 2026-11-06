"""Modified from ming024/FastSpeech2 ``transformer/Models.py``.

Changes:
* ``Encoder`` no longer owns the phoneme embedding (it lives in ``FastSpeech2``
  so the learned aligner can reuse the raw embeddings) and takes embedded input.
* Sinusoidal positions are computed on the fly and cached, so there is no
  ``max_seq_len`` limit (long Telugu utterances such as T4 are never truncated).
* Encoder and Decoder share one implementation (``FFTStack``); sizes come from
  ``config.yaml`` (encoder: 4 layers, decoder: 6 layers, 256-d, 2 heads).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .Layers import FFTBlock


def get_sinusoid_encoding_table(length: int, dim: int, device=None) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim, device=device)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div[: dim // 2])
    return table


class FFTStack(nn.Module):
    """Positional encoding + N FFT blocks."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_inner: int, kernel_sizes, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [FFTBlock(d_model, n_heads, d_inner, kernel_sizes, dropout) for _ in range(n_layers)]
        )
        self._pe_cache: torch.Tensor | None = None

    def _positions(self, length: int, device) -> torch.Tensor:
        if self._pe_cache is None or self._pe_cache.size(0) < length or self._pe_cache.device != device:
            self._pe_cache = get_sinusoid_encoding_table(max(length, 1024), self.d_model, device)
        return self._pe_cache[:length]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self._positions(x.size(1), x.device).unsqueeze(0).to(x.dtype)
        for layer in self.layers:
            x = layer(x, mask)
        return x

    @classmethod
    def from_config(cls, c: dict) -> "FFTStack":
        return cls(c["layers"], c["hidden"], c["heads"], c["conv_filter_size"], c["conv_kernel_size"], c["dropout"])


class Encoder(FFTStack):
    """Phoneme-side FFT stack (input: embedded phonemes [B, T_text, d])."""


class Decoder(FFTStack):
    """Frame-side FFT stack (input: length-regulated hidden states [B, T_mel, d])."""
