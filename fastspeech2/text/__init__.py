"""Language-specific text frontends (normalisation + G2P) and the token vocabulary."""

from __future__ import annotations

from .base import TextAnalysis, TextFrontend
from .symbols import Vocabulary


def get_frontend(language: str) -> TextFrontend:
    language = language.lower()
    if language in {"english", "en"}:
        from .english import EnglishFrontend

        return EnglishFrontend()
    if language in {"telugu", "te"}:
        from .telugu import TeluguFrontend

        return TeluguFrontend()
    raise ValueError(f"Unsupported language: {language!r} (expected 'english' or 'telugu')")


__all__ = ["TextAnalysis", "TextFrontend", "Vocabulary", "get_frontend"]
