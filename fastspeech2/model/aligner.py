"""NEW FILE (not in ming024/FastSpeech2). Unsupervised text-to-mel alignment, replacing MFA.

FastSpeech 2 needs phoneme durations. ming024 reads them from MFA TextGrids,
but MFA has no pretrained Telugu acoustic model or lexicon, and Svarah's
Indian-accented English is a poor fit for the US-English model. We therefore
learn the alignment jointly with the TTS model, following
"One TTS Alignment To Rule Them All" (Badlani et al., 2022 / RAD-TTS):

* a light conv encoder embeds text and mel; soft alignment = softmax of
  negative squared L2 distances, multiplied by a beta-binomial prior;
* ``forward_sum_loss`` (CTC) maximises the likelihood of all monotonic paths;
* Monotonic Alignment Search (Viterbi) turns the soft alignment into hard
  durations, used as duration targets and for the length regulator;
* ``bin_loss`` pulls the soft alignment towards the hard one.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(fn):
            return fn

        return wrap if not (args and callable(args[0])) else args[0]

    prange = range


# Large finite negative instead of -inf: -inf leads to NaN gradients in log_softmax/CTC.
NEG_INF = -1e4


class AlignmentEncoder(nn.Module):
    def __init__(self, n_mels: int, n_text: int, n_att: int = 80, temperature: float = 5e-4):
        super().__init__()
        self.temperature = temperature
        self.key_proj = nn.Sequential(
            nn.Conv1d(n_text, n_text * 2, 3, padding=1), nn.ReLU(), nn.Conv1d(n_text * 2, n_att, 1)
        )
        self.query_proj = nn.Sequential(
            nn.Conv1d(n_mels, n_mels * 2, 3, padding=1), nn.ReLU(),
            nn.Conv1d(n_mels * 2, n_mels, 1), nn.ReLU(), nn.Conv1d(n_mels, n_att, 1),
        )

    def forward(self, mels, text_emb, text_mask, attn_prior):
        """
        mels: [B, T_mel, n_mels]; text_emb: [B, T_text, n_text]; text_mask: [B, T_text] (True=pad)
        attn_prior: [B, T_mel, T_text]
        returns soft attention [B, 1, T_mel, T_text] and log-probabilities (same shape).
        """
        with torch.autocast(device_type=mels.device.type, enabled=False):
            keys = self.key_proj(text_emb.float().transpose(1, 2)).transpose(1, 2)
            queries = self.query_proj(mels.float().transpose(1, 2)).transpose(1, 2)
            # squared L2 distance without cdist's sqrt (its gradient is NaN at 0)
            dist = (queries.pow(2).sum(-1, keepdim=True) - 2 * queries @ keys.transpose(1, 2)
                    + keys.pow(2).sum(-1).unsqueeze(1)).clamp_min(0.0)  # [B, T_mel, T_text]
            logits = (-self.temperature * dist).masked_fill(text_mask.unsqueeze(1), NEG_INF)
            logp = F.log_softmax(logits, dim=-1) + torch.log(attn_prior.float() + 1e-8)
            logp = logp.masked_fill(text_mask.unsqueeze(1), NEG_INF)
            soft = torch.softmax(logp, dim=-1)
        return soft.unsqueeze(1), logp.unsqueeze(1)


@njit(cache=True)
def _mas_single(logp: np.ndarray) -> np.ndarray:
    t_mel, t_text = logp.shape
    acc = np.full((t_mel, t_text), -np.inf)
    back = np.zeros((t_mel, t_text), dtype=np.int64)
    acc[0, 0] = logp[0, 0]
    for i in range(1, t_mel):
        for j in range(min(i + 1, t_text)):
            stay = acc[i - 1, j]
            move = acc[i - 1, j - 1] if j > 0 else -np.inf
            if move > stay:
                acc[i, j] = move + logp[i, j]
                back[i, j] = j - 1
            else:
                acc[i, j] = stay + logp[i, j]
                back[i, j] = j
    path = np.zeros((t_mel, t_text), dtype=np.float32)
    j = t_text - 1
    for i in range(t_mel - 1, -1, -1):
        path[i, j] = 1.0
        if i > 0:
            j = back[i, j]
    return path


@njit(parallel=True, cache=True)
def _mas_batch(logp: np.ndarray, mel_lens: np.ndarray, text_lens: np.ndarray) -> np.ndarray:
    out = np.zeros(logp.shape, dtype=np.float32)
    for b in prange(logp.shape[0]):
        m, t = mel_lens[b], text_lens[b]
        out[b, :m, :t] = _mas_single(logp[b, :m, :t])
    return out


@torch.no_grad()
def binarize_attention(soft: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
    """Monotonic Alignment Search on log(soft) -> hard 0/1 alignment [B, 1, T_mel, T_text]."""
    logp = torch.log(soft.squeeze(1).float().clamp_min(1e-12)).cpu().numpy().astype(np.float64)
    hard = _mas_batch(logp, mel_lens.cpu().numpy().astype(np.int64), text_lens.cpu().numpy().astype(np.int64))
    return torch.from_numpy(hard).to(soft.device).unsqueeze(1)


def forward_sum_loss(attn_logprob: torch.Tensor, text_lens: torch.Tensor, mel_lens: torch.Tensor,
                     blank_logprob: float = -1.0) -> torch.Tensor:
    """Batched CTC over all monotonic alignments (target sequence 1..T_text)."""
    with torch.autocast(device_type=attn_logprob.device.type, enabled=False):
        lp = attn_logprob.squeeze(1).float()  # [B, T_mel, T_text]
        lp = F.pad(lp, (1, 0), value=blank_logprob)  # class 0 = blank
        classes = torch.arange(lp.size(-1), device=lp.device)
        lp = lp.masked_fill(classes.view(1, 1, -1) > text_lens.view(-1, 1, 1), NEG_INF)
        lp = F.log_softmax(lp, dim=-1)
        targets = torch.arange(1, lp.size(-1), device=lp.device).unsqueeze(0).expand(lp.size(0), -1)
        return F.ctc_loss(lp.transpose(0, 1), targets, mel_lens, text_lens, blank=0,
                          reduction="mean", zero_infinity=True)


def bin_loss(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    log_sum = torch.log(torch.clamp(soft[hard == 1], min=1e-12)).sum()
    return -log_sum / hard.sum().clamp_min(1.0)
