"""Modified from ming024/FastSpeech2 ``evaluate.py``.

Computes the average of every loss term over the whole validation set
(utterance-weighted, teacher-forced) and saves one figure per validation with
the target mel, predicted mel and the learned soft alignment, which is the
quickest way to see whether the aligner has converged.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .utils.misc import get_logger
from .utils.tools import to_device

LOSS_KEYS = ["total", "mel", "postnet", "duration", "pitch", "energy", "ctc", "bin"]
log = get_logger("evaluate")


@torch.no_grad()
def evaluate(model, loader, criterion, step: int, device, use_amp: bool = False,
             sample_dir: str | Path | None = None, language: str = "") -> dict:
    was_training = model.training
    model.eval()
    sums = {k: 0.0 for k in LOSS_KEYS}
    count, first = 0, None
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch)
            losses = criterion(out, batch, step)
        n = len(batch["ids"])
        for k in LOSS_KEYS:
            sums[k] += float(losses[k]) * n
        count += n
        if first is None:
            first = (batch, out)
    if sample_dir is not None and first is not None:
        _save_sample(*first, Path(sample_dir) / f"val_step_{step:07d}.png", f"{language} | step {step}")
    if was_training:
        model.train()
    return {k: v / max(1, count) for k, v in sums.items()}


def _save_sample(batch: dict, out: dict, path: Path, title: str) -> None:
    try:
        from .utils.plots import plot_validation_sample

        n_t, n_m = int(batch["text_lens"][0]), int(batch["mel_lens"][0])
        plot_validation_sample(
            batch["mels"][0, :n_m].float().cpu().numpy(),
            out["mel_post"][0, :n_m].float().cpu().numpy(),
            out["attn_soft"][0, 0, :n_m, :n_t].float().cpu().numpy(),
            path, title=f"{title} | {batch['ids'][0]}",
        )
    except Exception as exc:  # plotting must never kill training
        log.warning("Could not save validation figure: %s", exc)
