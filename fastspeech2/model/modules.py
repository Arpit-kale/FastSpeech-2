"""Modified from ming024/FastSpeech2 ``model/modules.py``.

Variance adaptor (paper §2.3): duration, pitch and energy predictors + length regulator.

Changes:
* Pitch/energy are always phoneme-level (the reference's ``phoneme_level``
  option); the targets are computed from the learned alignment (see
  ``model/fastspeech2.py``) instead of MFA durations.
* Bin edges come from ``stats.json`` and are stored as buffers, so a checkpoint
  is self-contained (no separate stats file at inference).
* ``p_transform`` lets inference scale F0 in **Hz** using the speaker's
  statistics (the reference multiplies the normalised value).
* Every real token receives at least one frame at inference, so short phones
  are never silently dropped.
* ``LengthRegulator`` is a function (``regulate_length``) using
  ``repeat_interleave``; a thin class wrapper is kept for API parity.
"""

from __future__ import annotations

import torch
from torch import nn


class VariancePredictor(nn.Module):
    """2 x (Conv1D -> ReLU -> LayerNorm -> Dropout) -> Linear, as in the paper."""

    def __init__(self, d_in: int, filters: int, kernel: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv1 = nn.Conv1d(d_in, filters, kernel, padding=pad)
        self.norm1 = nn.LayerNorm(filters)
        self.conv2 = nn.Conv1d(filters, filters, kernel, padding=pad)
        self.norm2 = nn.LayerNorm(filters)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(filters, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.conv1(x.transpose(1, 2))).transpose(1, 2)
        y = self.dropout(self.norm1(y))
        y = torch.relu(self.conv2(y.transpose(1, 2))).transpose(1, 2)
        y = self.dropout(self.norm2(y))
        return self.proj(y).squeeze(-1).masked_fill(mask, 0.0)


def regulate_length(x: torch.Tensor, durations: torch.Tensor, max_len: int | None = None):
    """Repeat each token vector ``durations[b, i]`` times. Returns (expanded, mel_lens)."""
    outs, lens = [], []
    for seq, dur in zip(x, durations):
        expanded = torch.repeat_interleave(seq, dur.long().clamp_min(0), dim=0)
        outs.append(expanded)
        lens.append(expanded.size(0))
    target = max_len or max(max(lens), 1)
    padded = x.new_zeros(len(outs), target, x.size(-1))
    for i, o in enumerate(outs):
        n = min(o.size(0), target)
        padded[i, :n] = o[:n]
    return padded, torch.tensor(lens, device=x.device, dtype=torch.long)


class VarianceAdaptor(nn.Module):
    def __init__(self, cfg: dict, stats: dict):
        super().__init__()
        d = cfg["encoder"]["hidden"]
        vp = cfg["variance_predictor"]
        n_bins = int(cfg["variance_embedding"]["n_bins"])
        self.duration_predictor = VariancePredictor(d, vp["filter_size"], vp["kernel_size"], vp["dropout"])
        self.pitch_predictor = VariancePredictor(d, vp["filter_size"], vp["kernel_size"], vp["dropout"])
        self.energy_predictor = VariancePredictor(d, vp["filter_size"], vp["kernel_size"], vp["dropout"])
        p_min, p_max = stats["pitch"]["min"], stats["pitch"]["max"]
        e_min, e_max = stats["energy"]["min"], stats["energy"]["max"]
        self.register_buffer("pitch_bins", torch.linspace(p_min, p_max, n_bins - 1))
        self.register_buffer("energy_bins", torch.linspace(e_min, e_max, n_bins - 1))
        self.pitch_embedding = nn.Embedding(n_bins, d)
        self.energy_embedding = nn.Embedding(n_bins, d)

    def forward(self, x, src_mask, dur_target=None, pitch_target=None, energy_target=None,
                d_control: float = 1.0, p_transform=None, e_control: float = 1.0):
        log_dur_pred = self.duration_predictor(x, src_mask)

        pitch_pred = self.pitch_predictor(x, src_mask)
        if pitch_target is not None:
            pitch_used = pitch_target
        else:
            pitch_used = p_transform(pitch_pred) if p_transform is not None else pitch_pred
        x = x + self.pitch_embedding(torch.bucketize(pitch_used, self.pitch_bins))

        energy_pred = self.energy_predictor(x, src_mask)
        if energy_target is not None:
            energy_used = energy_target
        else:
            energy_used = energy_pred * e_control
        x = x + self.energy_embedding(torch.bucketize(energy_used, self.energy_bins))

        if dur_target is not None:
            dur_used = dur_target
        else:
            dur_used = torch.clamp(torch.round((torch.exp(log_dur_pred) - 1.0) * d_control), min=0)
            dur_used = dur_used.masked_fill(src_mask, 0)
            # every real token gets at least one frame so it is never silently dropped
            dur_used = torch.where(src_mask, dur_used, dur_used.clamp_min(1))
        return {
            "x": x,
            "log_dur_pred": log_dur_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
            "durations": dur_used,
            "pitch_used": pitch_used,
            "energy_used": energy_used,
        }


class LengthRegulator(nn.Module):
    """API-compatible wrapper around :func:`regulate_length`."""

    def forward(self, x, duration, max_len=None):
        return regulate_length(x, duration, max_len)
