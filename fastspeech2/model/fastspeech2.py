"""Modified from ming024/FastSpeech2 ``model/fastspeech2.py``.

Changes:
* Integrated learned aligner (``model/aligner.py``) replaces MFA durations:
  the training ``forward`` computes a soft alignment, binarises it with
  Monotonic Alignment Search, and derives duration and phoneme-level
  pitch/energy targets from it on the fly.
* Length regulation during training is an exact matrix product with the hard
  alignment; inference uses the predicted durations.
* Separate ``forward`` (teacher-forced) and ``infer`` methods.
* The speaker embedding is used only when the dataset has >1 speaker (Svarah).
"""

from __future__ import annotations

import torch
from torch import nn

from ..transformer import Decoder, Encoder, PostNet
from ..utils.tools import get_mask_from_lengths as lengths_to_mask
from .aligner import AlignmentEncoder, binarize_attention
from .modules import VarianceAdaptor, regulate_length


class FastSpeech2(nn.Module):
    def __init__(self, model_cfg: dict, n_symbols: int, n_speakers: int, stats: dict, n_mels: int = 80):
        super().__init__()
        enc, dec = model_cfg["encoder"], model_cfg["decoder"]
        d = enc["hidden"]
        self.n_mels = n_mels
        self.embedding = nn.Embedding(n_symbols, d, padding_idx=0)
        self.encoder = Encoder.from_config(enc)
        self.variance_adaptor = VarianceAdaptor(model_cfg, stats)
        self.decoder = Decoder.from_config(dec)
        self.mel_linear = nn.Linear(dec["hidden"], n_mels)
        pn = model_cfg.get("postnet", {})
        self.postnet = PostNet(n_mels, pn.get("channels", 512), pn.get("kernel_size", 5), pn.get("layers", 5), pn.get("dropout", 0.5))
        self.speaker_embedding = nn.Embedding(n_speakers, d) if n_speakers > 1 else None
        al = model_cfg.get("aligner", {})
        self.aligner = AlignmentEncoder(n_mels, d, al.get("attention_channels", 80), al.get("temperature", 5e-4))

    # ------------------------------------------------------------------
    @staticmethod
    def _token_average(frame_values: torch.Tensor, hard: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """Average a frame-level signal over each token's aligned frames."""
        summed = torch.bmm(hard.squeeze(1).transpose(1, 2), frame_values.unsqueeze(-1)).squeeze(-1)
        return summed / durations.clamp_min(1.0)

    def _encode(self, tokens, text_lens, speakers):
        src_mask = lengths_to_mask(text_lens, tokens.size(1))
        emb = self.embedding(tokens)
        x = self.encoder(emb, src_mask)
        if self.speaker_embedding is not None:
            x = x + self.speaker_embedding(speakers).unsqueeze(1)
        return emb, x, src_mask

    def forward(self, batch: dict) -> dict:
        """Teacher-forced pass used for training/validation."""
        tokens, text_lens = batch["tokens"], batch["text_lens"]
        mels, mel_lens = batch["mels"], batch["mel_lens"]
        emb, x, src_mask = self._encode(tokens, text_lens, batch["speakers"])
        mel_mask = lengths_to_mask(mel_lens, mels.size(1))

        attn_soft, attn_logprob = self.aligner(mels, emb, src_mask, batch["attn_prior"])
        attn_hard = binarize_attention(attn_soft, text_lens, mel_lens)
        durations = attn_hard.sum(2).squeeze(1)  # [B, T_text]
        pitch_tgt = self._token_average(batch["pitch"], attn_hard, durations).masked_fill(src_mask, 0.0)
        energy_tgt = self._token_average(batch["energy"], attn_hard, durations).masked_fill(src_mask, 0.0)

        va = self.variance_adaptor(x, src_mask, dur_target=durations, pitch_target=pitch_tgt, energy_target=energy_tgt)
        # exact length regulation through the hard alignment matrix
        expanded = torch.bmm(attn_hard.squeeze(1).to(va["x"].dtype), va["x"])
        out = self.decoder(expanded, mel_mask)
        mel = self.mel_linear(out)
        mel_post = mel + self.postnet(mel)
        return {
            "mel": mel, "mel_post": mel_post, "mel_mask": mel_mask, "src_mask": src_mask,
            "log_dur_pred": va["log_dur_pred"], "pitch_pred": va["pitch_pred"], "energy_pred": va["energy_pred"],
            "dur_target": durations, "pitch_target": pitch_tgt, "energy_target": energy_tgt,
            "attn_soft": attn_soft, "attn_logprob": attn_logprob, "attn_hard": attn_hard,
        }

    @torch.no_grad()
    def infer(self, tokens, speakers, d_control=1.0, p_transform=None, e_control=1.0) -> dict:
        text_lens = torch.tensor([tokens.size(1)] * tokens.size(0), device=tokens.device)
        _, x, src_mask = self._encode(tokens, text_lens, speakers)
        va = self.variance_adaptor(x, src_mask, d_control=d_control, p_transform=p_transform, e_control=e_control)
        expanded, mel_lens = regulate_length(va["x"], va["durations"])
        mel_mask = lengths_to_mask(mel_lens, expanded.size(1))
        out = self.decoder(expanded, mel_mask)
        mel = self.mel_linear(out)
        mel_post = mel + self.postnet(mel)
        return {
            "mel_post": mel_post, "mel_lens": mel_lens, "durations": va["durations"],
            "log_dur_pred": va["log_dur_pred"], "pitch_pred": va["pitch_used"], "energy_pred": va["energy_used"],
        }
