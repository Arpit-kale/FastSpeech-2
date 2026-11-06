"""Modified from ming024/FastSpeech2 ``utils/tools.py``.

Kept: ``get_mask_from_lengths``, ``to_device``, ``pad_1D``/``pad_2D`` (same
names and semantics, True = padded position). Plotting moved to
``utils/plots.py``; logging to CSV/TensorBoard lives in ``train.py``.
"""

from __future__ import annotations

import numpy as np
import torch


def get_mask_from_lengths(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Boolean padding mask [B, max_len], True on padded positions."""
    max_len = int(max_len or lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ids >= lengths.unsqueeze(1)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def pad_1D(seqs, value=0.0, dtype=np.float32) -> np.ndarray:
    n = max(len(s) for s in seqs)
    out = np.full((len(seqs), n), value, dtype=dtype)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def pad_2D(seqs, value=0.0) -> np.ndarray:
    n = max(len(s) for s in seqs)
    out = np.full((len(seqs), n, seqs[0].shape[1]), value, dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out
