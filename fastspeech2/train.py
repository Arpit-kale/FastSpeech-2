"""Modified from ming024/FastSpeech2 ``train.py``.

Step-based trainer. Changes from the reference loop:
* frame-budget bucketed batches (``dataset.FrameBudgetBatchSampler``) instead of
  a fixed batch size with group sorting;
* AdamW + Noam ``LambdaLR`` (``model/optimizer.py``), AMP fp16 with loss scaling,
  gradient clipping and accumulation;
* automatic resume from ``checkpoints/last.pt`` (model, optimiser, scheduler,
  scaler, step), rolling ``step_*.pt`` files, ``best.pt`` by validation loss,
  optional early stopping, and a divergence guard;
* CSV logs (used for the loss plots and the report) + TensorBoard;
* exports the self-contained ``model_inference.pt`` at the end.
Validation is in ``evaluate.py`` as in the reference.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import FrameBudgetBatchSampler, TTSDataset, collate
from .evaluate import LOSS_KEYS, evaluate
from .model import FastSpeech2Loss
from .model.optimizer import build_optimizer
from .text import Vocabulary
from .utils import ensure_dir, get_device, get_logger, seed_everything, to_plain
from .utils.misc import add_file_handler
from .utils.model import build_model, export_for_inference, save_checkpoint
from .utils.tools import to_device


class _CsvLog:
    def __init__(self, path: Path, fields: list[str], append: bool):
        self.path = path
        self.fields = fields
        new = not (append and path.exists())
        self.fh = open(path, "a" if not new else "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fh, fieldnames=fields)
        if new:
            self.writer.writeheader()

    def write(self, row: dict) -> None:
        self.writer.writerow({k: row.get(k, "") for k in self.fields})
        self.fh.flush()


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        t = cfg.train
        seed_everything(int(t.get("seed", 1234)))
        self.device = get_device(t.get("device", "auto"))
        self.ckpt_dir = ensure_dir(cfg.paths.checkpoint_dir)
        self.log_dir = ensure_dir(cfg.paths.log_dir)
        self.log = get_logger(f"train.{cfg.language}")
        add_file_handler(self.log, self.log_dir / "train.log")

        pre = Path(cfg.paths.preprocessed_dir)
        self.stats = json.loads((pre / "stats.json").read_text())
        self.speaker_map = json.loads((pre / "speakers.json").read_text())
        self.vocab = Vocabulary.load(pre / "vocab.json")
        self.train_set = TTSDataset(pre, "train", self.vocab)
        self.val_set = TTSDataset(pre, "val", self.vocab)
        prior_scale = float(cfg.model.get("aligner", {}).get("prior_scaling", 1.0))
        collate_fn = partial(collate, prior_scaling=prior_scale)
        workers = int(t.get("num_workers", 2))
        self.sampler = FrameBudgetBatchSampler(self.train_set.lengths, int(t.max_frames_per_batch),
                                               int(t.max_batch_size), shuffle=True, seed=int(t.get("seed", 1234)))
        self.train_loader = DataLoader(self.train_set, batch_sampler=self.sampler, collate_fn=collate_fn,
                                       num_workers=workers, pin_memory=self.device.type == "cuda",
                                       persistent_workers=workers > 0)
        val_sampler = FrameBudgetBatchSampler(self.val_set.lengths, int(t.max_frames_per_batch),
                                              int(t.max_batch_size), shuffle=False)
        self.val_loader = DataLoader(self.val_set, batch_sampler=val_sampler, collate_fn=collate_fn,
                                     num_workers=0)

        self.model = build_model(cfg.model, len(self.vocab), max(1, len(self.speaker_map)),
                                 self.stats, int(cfg.audio.n_mels)).to(self.device)
        self.criterion = FastSpeech2Loss(to_plain(cfg.model) | {"loss_weights": to_plain(cfg.get("loss_weights", {}))})
        self.optimizer, self.scheduler = build_optimizer(self.model, to_plain(cfg.optimizer))
        self.use_amp = bool(t.get("fp16", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.step, self.best_val, self.bad_vals = 0, float("inf"), 0
        if t.get("resume", True):
            self._try_resume()
        self.train_csv = _CsvLog(self.log_dir / "train_log.csv", ["step", *LOSS_KEYS, "lr", "sec_per_step"], self.step > 0)
        self.val_csv = _CsvLog(self.log_dir / "val_log.csv", ["step", *LOSS_KEYS], self.step > 0)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(self.log_dir / "tensorboard"))
        except Exception:  # tensorboard optional
            self.tb = None

        n_params = sum(p.numel() for p in self.model.parameters())
        self.log.info("Device %s | params %.2fM | train %d utt (%d batches) | val %d utt | speakers %d | vocab %d",
                      self.device, n_params / 1e6, len(self.train_set), len(self.sampler), len(self.val_set),
                      len(self.speaker_map), len(self.vocab))

    # ------------------------------------------------------------------
    def _meta(self) -> dict:
        return {
            "language": self.cfg.language,
            "model_cfg": to_plain(self.cfg.model),
            "audio_cfg": to_plain(self.cfg.audio),
            "vocab": self.vocab.symbols,
            "speaker_map": self.speaker_map,
            "stats": self.stats,
        }

    def _try_resume(self) -> None:
        last = self.ckpt_dir / "last.pt"
        if not last.exists():
            return
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        if self.scaler is not None and ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
        self.step = int(ckpt["step"])
        self.best_val = float(ckpt.get("best_val", float("inf")))
        self.log.info("Resumed from %s at step %d (best val %.4f)", last, self.step, self.best_val)

    def _to_device(self, batch: dict) -> dict:
        return to_device(batch, self.device)

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp)

    # ------------------------------------------------------------------
    def fit(self) -> Path:
        t = self.cfg.train
        total = int(t.total_steps)
        accum = max(1, int(t.get("grad_accum", 1)))
        clip = float(t.get("grad_clip", 1.0))
        log_every, val_every, save_every = int(t.log_every), int(t.val_every), int(t.save_every)
        patience = int(t.get("early_stopping_patience", 0))

        self.model.train()
        running = {k: 0.0 for k in LOSS_KEYS}
        n_running, epoch, t0, bad_batches = 0, 0, time.time(), 0
        done = self.step >= total
        while not done:
            self.sampler.set_epoch(epoch)
            for i, batch in enumerate(self.train_loader):
                batch = self._to_device(batch)
                with self._autocast():
                    out = self.model(batch)
                    losses = self.criterion(out, batch, self.step)
                loss = losses["total"] / accum
                if not torch.isfinite(loss):
                    bad_batches += 1
                    self.log.warning("Non-finite loss at step %d; batch skipped (%d in a row).", self.step, bad_batches)
                    self.optimizer.zero_grad(set_to_none=True)
                    if bad_batches >= 20:
                        raise RuntimeError("Training diverged (20 consecutive non-finite losses). "
                                           "Lower optimizer.peak_lr or disable fp16.")
                    continue
                bad_batches = 0
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                for k in LOSS_KEYS:
                    running[k] += float(losses[k].detach())
                n_running += 1
                if (i + 1) % accum:
                    continue

                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.step += 1

                if self.step % log_every == 0 or self.step == 1:
                    avg = {k: running[k] / max(1, n_running) for k in LOSS_KEYS}
                    sps = (time.time() - t0) / max(1, n_running / accum)
                    lr = self.scheduler.get_last_lr()[0]
                    self.train_csv.write({"step": self.step, **{k: f"{v:.5f}" for k, v in avg.items()},
                                          "lr": f"{lr:.3e}", "sec_per_step": f"{sps:.3f}"})
                    self._tb("train", avg, lr)
                    self.log.info("step %6d/%d | total %.4f mel %.4f post %.4f dur %.4f pitch %.4f energy %.4f "
                                  "ctc %.4f bin %.4f | lr %.2e | %.2fs/step", self.step, total, avg["total"],
                                  avg["mel"], avg["postnet"], avg["duration"], avg["pitch"], avg["energy"],
                                  avg["ctc"], avg["bin"], lr, sps)
                    running = {k: 0.0 for k in LOSS_KEYS}
                    n_running, t0 = 0, time.time()

                if self.step % val_every == 0 or self.step == total:
                    val = self.validate()
                    improved = val["total"] < self.best_val
                    if improved:
                        self.best_val, self.bad_vals = val["total"], 0
                        save_checkpoint(self.ckpt_dir / "best.pt", self.model, self.optimizer, self.scheduler,
                                        self.scaler, self.step, self._meta(), self.best_val)
                    else:
                        self.bad_vals += 1
                    self.log.info("VAL step %d | total %.4f mel %.4f post %.4f dur %.4f pitch %.4f energy %.4f%s",
                                  self.step, val["total"], val["mel"], val["postnet"], val["duration"],
                                  val["pitch"], val["energy"], "  (best)" if improved else "")
                    if patience and self.bad_vals >= patience:
                        self.log.info("Early stopping: no improvement for %d validations.", patience)
                        done = True

                if self.step % save_every == 0 or self.step == total or done:
                    self._save_periodic()
                if self.step >= total or done:
                    done = True
                    break
            epoch += 1

        best = self.ckpt_dir / "best.pt"
        if not best.exists():
            shutil.copy(self.ckpt_dir / "last.pt", best)
        exported = export_for_inference(best, self.ckpt_dir / "model_inference.pt")
        self.log.info("Training finished at step %d; best val %.4f", self.step, self.best_val)
        return exported

    def _save_periodic(self) -> None:
        path = self.ckpt_dir / f"step_{self.step:07d}.pt"
        save_checkpoint(path, self.model, self.optimizer, self.scheduler, self.scaler, self.step,
                        self._meta(), self.best_val)
        shutil.copy(path, self.ckpt_dir / "last.pt")
        keep = int(self.cfg.train.get("keep_last", 3))
        olds = sorted(self.ckpt_dir.glob("step_*.pt"))
        for p in olds[:-keep] if keep > 0 else []:
            p.unlink(missing_ok=True)

    def _tb(self, tag: str, values: dict, lr: float | None = None) -> None:
        if self.tb is None:
            return
        for k, v in values.items():
            self.tb.add_scalar(f"{tag}/{k}", v, self.step)
        if lr is not None:
            self.tb.add_scalar("train/lr", lr, self.step)

    def validate(self) -> dict:
        avg = evaluate(self.model, self.val_loader, self.criterion, self.step, self.device, self.use_amp,
                       self.log_dir / "samples", self.cfg.language)
        self.val_csv.write({"step": self.step, **{k: f"{v:.5f}" for k, v in avg.items()}})
        self._tb("val", avg)
        return avg
