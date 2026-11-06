"""Modified from ming024/FastSpeech2 ``transformer/Layers.py``.

Changes:
* ``FFTBlock`` uses ``nn.MultiheadAttention``; padded positions are zeroed
  after each sub-layer exactly as in the reference.
* ``PostNet`` keeps the 5 x (Conv1D 512, k=5, BatchNorm, tanh) design; ``ConvNorm``
  was folded into it (plain ``nn.Conv1d``), and the sizes come from the config.
"""

from __future__ import annotations

import torch
from torch import nn

from .SubLayers import PositionwiseFeedForward


class FFTBlock(nn.Module):
    """Feed-Forward Transformer block: self-attention + conv FFN, each with residual + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_inner: int, kernel_sizes, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_inner, tuple(kernel_sizes), dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out)).masked_fill(mask.unsqueeze(-1), 0.0)
        x = self.norm2(x + self.ffn(x)).masked_fill(mask.unsqueeze(-1), 0.0)
        return x


class PostNet(nn.Module):
    """Five 1D conv layers with batch norm (Tacotron 2 style residual refiner)."""

    def __init__(self, n_mels: int = 80, channels: int = 512, kernel: int = 5, n_layers: int = 5, dropout: float = 0.5):
        super().__init__()
        layers = []
        for i in range(n_layers):
            c_in = n_mels if i == 0 else channels
            c_out = n_mels if i == n_layers - 1 else channels
            block = [nn.Conv1d(c_in, c_out, kernel, padding=(kernel - 1) // 2), nn.BatchNorm1d(c_out)]
            if i < n_layers - 1:
                block.append(nn.Tanh())
            block.append(nn.Dropout(dropout))
            layers.append(nn.Sequential(*block))
        self.layers = nn.ModuleList(layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)
