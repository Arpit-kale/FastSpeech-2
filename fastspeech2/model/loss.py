"""Modified from ming024/FastSpeech2 ``model/loss.py``.

Training objective: mel L1 (before/after PostNet) + MSE on log-duration,
phoneme-level pitch and energy (as in the reference) + two new alignment
terms: CTC forward-sum and a binarisation loss ramped in after
``bin_loss_start_step``. All terms are weighted from ``loss_weights``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .aligner import bin_loss, forward_sum_loss


class FastSpeech2Loss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.w = cfg.get("loss_weights", {})
        al = cfg.get("aligner", {})
        self.bin_start = int(al.get("bin_loss_start_step", 2000))
        self.bin_warmup = int(al.get("bin_loss_warmup_steps", 2000))

    def bin_weight(self, step: int) -> float:
        if step < self.bin_start:
            return 0.0
        return min(1.0, (step - self.bin_start) / max(1, self.bin_warmup))

    def forward(self, out: dict, batch: dict, step: int) -> dict[str, torch.Tensor]:
        mel_valid = ~out["mel_mask"]
        src_valid = ~out["src_mask"]
        target = batch["mels"]

        mel_loss = F.l1_loss(out["mel"][mel_valid].float(), target[mel_valid])
        post_loss = F.l1_loss(out["mel_post"][mel_valid].float(), target[mel_valid])
        log_dur_target = torch.log(out["dur_target"].float() + 1.0)
        dur_loss = F.mse_loss(out["log_dur_pred"][src_valid].float(), log_dur_target[src_valid])
        pitch_loss = F.mse_loss(out["pitch_pred"][src_valid].float(), out["pitch_target"][src_valid])
        energy_loss = F.mse_loss(out["energy_pred"][src_valid].float(), out["energy_target"][src_valid])
        ctc = forward_sum_loss(out["attn_logprob"], batch["text_lens"], batch["mel_lens"])
        b_w = self.bin_weight(step)
        b_loss = bin_loss(out["attn_hard"], out["attn_soft"])

        total = (
            self.w.get("mel", 1.0) * mel_loss
            + self.w.get("postnet", 1.0) * post_loss
            + self.w.get("duration", 1.0) * dur_loss
            + self.w.get("pitch", 1.0) * pitch_loss
            + self.w.get("energy", 1.0) * energy_loss
            + self.w.get("ctc", 1.0) * ctc
            + self.w.get("bin", 1.0) * b_w * b_loss
        )
        return {
            "total": total, "mel": mel_loss, "postnet": post_loss, "duration": dur_loss,
            "pitch": pitch_loss, "energy": energy_loss, "ctc": ctc, "bin": b_loss,
        }
