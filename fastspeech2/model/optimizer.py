"""Modified from ming024/FastSpeech2 ``model/optimizer.py``.

Changes: ``ScheduledOptim`` (Adam + manual Noam LR) is replaced by AdamW with a
``LambdaLR`` Noam schedule, so the scheduler state is saved/resumed by PyTorch
and the peak learning rate is an explicit config value
(lr(step) = peak_lr * min(step / warmup, sqrt(warmup / step))).
"""

from __future__ import annotations

import math

import torch


def noam_lambda(warmup: int):
    def fn(step: int) -> float:
        step = max(step, 1)
        return min(step / warmup, math.sqrt(warmup / step))

    return fn


def build_optimizer(model: torch.nn.Module, cfg: dict):
    """Return (optimizer, scheduler) from the ``optimizer`` config section."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["peak_lr"]), betas=tuple(cfg.get("betas", [0.9, 0.98])),
        eps=float(cfg.get("eps", 1e-9)), weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, noam_lambda(int(cfg["warmup_steps"])))
    return optimizer, scheduler
