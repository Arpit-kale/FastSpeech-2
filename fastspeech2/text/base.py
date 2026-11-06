"""NEW FILE (not in ming024/FastSpeech2): common frontend interface.

Common interface for text frontends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

PAD = "_"
SP = "sp"  # short pause (comma-like punctuation)
SIL = "sil"  # sentence-final / long pause
SPECIAL_SYMBOLS = [PAD, SP, SIL]

SHORT_PAUSE_PUNCT = set(",;:-–—")
LONG_PAUSE_PUNCT = set(".!?।॥")


@dataclass
class WordPron:
    word: str
    phones: list[str]


@dataclass
class TextAnalysis:
    """Everything the frontend knows about one input string."""

    original: str
    normalized: str
    words: list[WordPron] = field(default_factory=list)

    @property
    def phonemes(self) -> list[str]:
        return [p for w in self.words for p in w.phones]

    def word_spans(self) -> list[tuple[str, int, int]]:
        """(word, start_token, end_token_exclusive) for plotting/analysis."""
        spans, idx = [], 0
        for w in self.words:
            spans.append((w.word, idx, idx + len(w.phones)))
            idx += len(w.phones)
        return spans

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phonemes"] = self.phonemes
        return d


class TextFrontend(ABC):
    """Normalise raw text and convert it to a phoneme sequence."""

    language: str = "base"

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """Full phone inventory (without special symbols)."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Expand numbers, dates, times, abbreviations; clean punctuation."""

    @abstractmethod
    def _g2p(self, normalized: str) -> list[WordPron]:
        """Phonemise normalised text, returning per-word pronunciations."""

    def is_valid_text(self, text: str) -> bool:
        """Cheap filter used during data selection."""
        return bool(text and text.strip())

    def analyze(self, text: str) -> TextAnalysis:
        normalized = self.normalize(text)
        words = self._finalize(self._g2p(normalized))
        return TextAnalysis(original=text, normalized=normalized, words=words)

    def phonemize(self, text: str) -> list[str]:
        return self.analyze(text).phonemes

    @staticmethod
    def _finalize(words: list[WordPron]) -> list[WordPron]:
        """Collapse repeated pauses, drop leading pauses, force one final ``sil``.

        Ending every utterance with exactly one ``sil`` keeps training data
        (which may lack punctuation) and inference inputs consistent.
        """
        out: list[WordPron] = []
        for w in words:
            if not w.phones:
                continue
            is_pause = w.phones[0] in (SP, SIL) and len(w.phones) == 1
            if is_pause:
                if not out:
                    continue
                prev = out[-1]
                if len(prev.phones) == 1 and prev.phones[0] in (SP, SIL):
                    if w.phones[0] == SIL:
                        out[-1] = w
                    continue
            out.append(w)
        while out and len(out[-1].phones) == 1 and out[-1].phones[0] in (SP, SIL):
            out.pop()
        out.append(WordPron(word=".", phones=[SIL]))
        return out
