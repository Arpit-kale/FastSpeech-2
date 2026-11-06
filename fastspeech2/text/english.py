"""Modified from ming024/FastSpeech2 ``synthesize.py::preprocess_english`` + ``text/cmudict.py`` (g2p_en on whole sentences for POS-aware homographs).

English frontend: rule-based normalisation + g2p_en (CMUdict, POS-aware homographs, neural OOV)."""

from __future__ import annotations

import re
import threading

from .base import LONG_PAUSE_PUNCT, SHORT_PAUSE_PUNCT, SIL, SP, TextFrontend, WordPron
from .numbers import normalize_english

_VOWELS = [
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
]
_CONSONANTS = [
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R",
    "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
]
ARPABET = _CONSONANTS + _VOWELS + [v + s for v in _VOWELS for s in "012"]

_NLTK_PACKAGES = [
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("corpora/cmudict", "cmudict"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]


def ensure_nltk_data() -> None:
    """Download the small NLTK resources g2p_en needs (no-op when present)."""
    import nltk

    for resource, package in _NLTK_PACKAGES:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


class EnglishFrontend(TextFrontend):
    language = "english"
    _g2p_model = None
    _lock = threading.Lock()

    @property
    def symbols(self) -> list[str]:
        return list(ARPABET)

    def normalize(self, text: str) -> str:
        return normalize_english(text)

    def is_valid_text(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        letters = sum(ch.isascii() and ch.isalpha() for ch in text)
        return letters / max(1, sum(not ch.isspace() for ch in text)) > 0.6

    @classmethod
    def _model(cls):
        if cls._g2p_model is None:
            with cls._lock:
                if cls._g2p_model is None:
                    ensure_nltk_data()
                    from g2p_en import G2p

                    cls._g2p_model = G2p()
        return cls._g2p_model

    def _g2p(self, normalized: str) -> list[WordPron]:
        """Run g2p_en on the whole sentence (so POS context can disambiguate
        homographs such as *conduct* N vs V) and split the output per word."""
        g2p = self._model()
        with self._lock:
            raw = g2p(normalized)
        groups: list[list[str]] = [[]]
        for tok in raw:
            if tok == " ":
                groups.append([])
            else:
                groups[-1].append(tok)
        tokens = self._tokenize_like_g2p(normalized)
        merged: list[str] = []  # g2p_en attaches clitics ('s, n't, 'll) to the previous word
        for tok in tokens:
            if merged and tok.startswith("'") or (merged and tok == "n't"):
                merged[-1] += tok
            else:
                merged.append(tok)
        if len(merged) == len(groups):
            words = merged
        elif len(tokens) == len(groups):
            words = tokens
        else:  # rare tokenisation mismatch; fall back to phone labels
            words = [" ".join(g) for g in groups]

        out: list[WordPron] = []
        for word, phones in zip(words, groups):
            if not phones:
                continue
            if all(not re.search(r"[A-Z]", p) for p in phones):
                mark = phones[0]
                if mark and mark[0] in LONG_PAUSE_PUNCT:
                    out.append(WordPron(word=mark, phones=[SIL]))
                elif mark and mark[0] in SHORT_PAUSE_PUNCT:
                    out.append(WordPron(word=mark, phones=[SP]))
                continue
            clean = [p for p in phones if p in self._symbol_set]
            if clean:
                out.append(WordPron(word=word, phones=clean))
        return out

    @property
    def _symbol_set(self) -> set[str]:
        return set(ARPABET)

    @staticmethod
    def _tokenize_like_g2p(text: str) -> list[str]:
        """Replicates g2p_en's own preprocessing + tokenisation to recover word labels."""
        import unicodedata

        from nltk import word_tokenize

        t = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
        t = t.lower()
        t = re.sub(r"[^ a-z'.,?!\-]", "", t)
        t = t.replace("i.e.", "that is").replace("e.g.", "for example")
        return word_tokenize(t)
