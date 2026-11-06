"""Modified from ming024/FastSpeech2 ``text/symbols.py`` + ``text/__init__.py`` (text_to_sequence) — per-language symbol table stored in checkpoints.

Symbol <-> id mapping that is stored inside every checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from .base import PAD, SPECIAL_SYMBOLS


class Vocabulary:
    def __init__(self, symbols: list[str]):
        ordered = list(SPECIAL_SYMBOLS) + [s for s in symbols if s not in SPECIAL_SYMBOLS]
        self.symbols = ordered
        self._index = {s: i for i, s in enumerate(ordered)}

    @classmethod
    def from_frontend(cls, frontend) -> "Vocabulary":
        return cls(frontend.symbols)

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._index

    @property
    def pad_id(self) -> int:
        return self._index[PAD]

    def encode(self, phonemes: list[str], strict: bool = False) -> list[int]:
        ids = []
        for p in phonemes:
            if p in self._index:
                ids.append(self._index[p])
            elif strict:
                raise KeyError(f"Unknown symbol {p!r}")
        return ids

    def decode(self, ids) -> list[str]:
        return [self.symbols[int(i)] for i in ids]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.symbols, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
