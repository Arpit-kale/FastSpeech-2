"""Modified from ming024/FastSpeech2 ``dataset.py`` (frame-budget batching, alignment prior in collate, frame-level pitch/energy).

PyTorch dataset, frame-budget batch sampler and collate function."""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.stats import betabinom
from torch.utils.data import Dataset, Sampler

from .text import Vocabulary


@lru_cache(maxsize=8192)
def beta_binomial_prior(n_text: int, n_mel: int, scaling: float = 1.0) -> np.ndarray:
    """Alignment prior [n_mel, n_text] that favours a near-diagonal path (RAD-TTS)."""
    x = np.arange(n_text)[None, :]
    i = np.arange(1, n_mel + 1)[:, None]
    a, b = scaling * i, scaling * (n_mel + 1 - i)
    return betabinom.pmf(x, n_text, a, b).astype(np.float32)


class TTSDataset(Dataset):
    def __init__(self, preprocessed_dir: str | Path, split: str, vocab: Vocabulary | None = None):
        self.root = Path(preprocessed_dir)
        self.vocab = vocab or Vocabulary.load(self.root / "vocab.json")
        self.speaker_map = json.loads((self.root / "speakers.json").read_text(encoding="utf-8"))
        self.items = []
        for line in (self.root / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            uid, spk, phones, text = line.split("|", 3)
            self.items.append({"id": uid, "speaker": spk, "phones": phones.split(), "text": text})
        self._lengths = None

    def __len__(self) -> int:
        return len(self.items)

    @property
    def lengths(self) -> list[int]:
        """Mel length of each item (read lazily from the .npy header)."""
        if self._lengths is None:
            lens = []
            for it in self.items:
                arr = np.load(self.root / "mel" / f"{it['id']}.npy", mmap_mode="r")
                lens.append(int(arr.shape[0]))
            self._lengths = lens
        return self._lengths

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        uid = it["id"]
        mel = np.load(self.root / "mel" / f"{uid}.npy")
        pitch = np.load(self.root / "pitch" / f"{uid}.npy")
        energy = np.load(self.root / "energy" / f"{uid}.npy")
        tokens = np.array(self.vocab.encode(it["phones"]), dtype=np.int64)
        n = min(len(mel), len(pitch), len(energy))
        return {
            "id": uid,
            "speaker": self.speaker_map.get(it["speaker"], 0),
            "tokens": tokens,
            "mel": mel[:n],
            "pitch": pitch[:n],
            "energy": energy[:n],
        }


class FrameBudgetBatchSampler(Sampler[list[int]]):
    """Groups utterances of similar length; each batch holds at most
    ``max_frames`` padded mel frames and ``max_items`` utterances."""

    def __init__(self, lengths: list[int], max_frames: int, max_items: int, shuffle: bool = True, seed: int = 0):
        self.lengths = lengths
        self.max_frames = max_frames
        self.max_items = max_items
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._batches = self._build(random.Random(seed))

    def _build(self, rng: random.Random) -> list[list[int]]:
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i] + (rng.random() if self.shuffle else 0))
        batches, cur, cur_max = [], [], 0
        for i in order:
            longest = max(cur_max, self.lengths[i])
            if cur and (longest * (len(cur) + 1) > self.max_frames or len(cur) >= self.max_items):
                batches.append(cur)
                cur, longest = [], self.lengths[i]
            cur.append(i)
            cur_max = longest
        if cur:
            batches.append(cur)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = self._build(rng) if self.shuffle else self._batches
        if self.shuffle:
            rng.shuffle(batches)
        self._batches = batches
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches)


def _pad_1d(seqs, value=0.0, dtype=np.float32):
    n = max(len(s) for s in seqs)
    out = np.full((len(seqs), n), value, dtype=dtype)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def _pad_2d(seqs, value=0.0):
    n = max(len(s) for s in seqs)
    out = np.full((len(seqs), n, seqs[0].shape[1]), value, dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def collate(batch: list[dict], prior_scaling: float = 1.0) -> dict:
    text_lens = np.array([len(b["tokens"]) for b in batch], dtype=np.int64)
    mel_lens = np.array([len(b["mel"]) for b in batch], dtype=np.int64)
    max_t, max_m = int(text_lens.max()), int(mel_lens.max())
    prior = np.zeros((len(batch), max_m, max_t), dtype=np.float32)
    for i, b in enumerate(batch):
        prior[i, : mel_lens[i], : text_lens[i]] = beta_binomial_prior(int(text_lens[i]), int(mel_lens[i]), prior_scaling)
    return {
        "ids": [b["id"] for b in batch],
        "speakers": torch.tensor([b["speaker"] for b in batch], dtype=torch.long),
        "tokens": torch.from_numpy(_pad_1d([b["tokens"] for b in batch], 0, np.int64)),
        "text_lens": torch.from_numpy(text_lens),
        "mels": torch.from_numpy(_pad_2d([b["mel"] for b in batch], 0.0)),
        "mel_lens": torch.from_numpy(mel_lens),
        "pitch": torch.from_numpy(_pad_1d([b["pitch"] for b in batch])),
        "energy": torch.from_numpy(_pad_1d([b["energy"] for b in batch])),
        "attn_prior": torch.from_numpy(prior),
    }
