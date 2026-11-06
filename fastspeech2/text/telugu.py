"""NEW FILE (not in ming024/FastSpeech2): Telugu grapheme-to-phoneme.

Rule-based Telugu grapheme-to-phoneme conversion.

Telugu orthography is close to phonemic, so an explicit akshara parser gives a
reliable phone sequence without a pronunciation lexicon (none of the reference
tools ship one for Telugu):

* consonant + (matra | virama | nothing) -> consonant + (vowel | none | inherent 'a').
  Telugu has no schwa deletion, so the inherent 'a' is always realised.
* anusvara (ం) -> homorganic nasal before a stop (ng/ny/N/n/m), 'm' elsewhere
  (e.g. word-final సమావేశం -> ...sh a m).
* visarga (ః) -> h, candrabindu -> nasal like anusvara.
* gemination comes for free (పట్టు -> p a T T u).
Retroflex consonants use capital letters (T, D, N, L, Sh) to keep them distinct
from the dentals, and all phones are ASCII so plots render without Telugu fonts.
"""

from __future__ import annotations

import re

from .base import LONG_PAUSE_PUNCT, SHORT_PAUSE_PUNCT, SIL, SP, TextFrontend, WordPron
from .numbers_te import normalize_telugu

VIRAMA = "్"
ANUSVARA = "ం"
CANDRABINDU = "ఁ"
VISARGA = "ః"
NUKTA = "఼"
AI_LENGTH_MARK = "ౖ"
LENGTH_MARK = "ౕ"

INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఌ": ["l", "u"], "ౡ": ["l", "uu"],
    "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"], "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}
MATRAS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"], "ృ": ["r", "u"],
    "ౄ": ["r", "uu"], "ె": ["e"], "ే": ["ee"], "ై": ["ai"], "ొ": ["o"], "ో": ["oo"],
    "ౌ": ["au"], "ౢ": ["l", "u"], "ౣ": ["l", "uu"],
}
CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "T", "ఠ": "Th", "డ": "D", "ఢ": "Dh", "ణ": "N",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "rr", "ల": "l", "ళ": "L", "ఴ": "zh", "వ": "v",
    "శ": "sh", "ష": "Sh", "స": "s", "హ": "h",
    "ౘ": "ts", "ౙ": "dz", "ౚ": "rr",
}
_HOMORGANIC = {
    **dict.fromkeys(["k", "kh", "g", "gh"], "ng"),
    **dict.fromkeys(["ch", "chh", "j", "jh"], "ny"),
    **dict.fromkeys(["T", "Th", "D", "Dh"], "N"),
    **dict.fromkeys(["t", "th", "d", "dh"], "n"),
    **dict.fromkeys(["p", "ph", "b", "bh"], "m"),
}
_NASAL_PLACEHOLDER = "<NASAL>"

VOWEL_PHONES = ["a", "aa", "i", "ii", "u", "uu", "e", "ee", "ai", "o", "oo", "au"]
CONSONANT_PHONES = sorted(set(CONSONANTS.values()) | {"ng", "ny", "N", "n", "m", "h"})
TELUGU_PHONES = VOWEL_PHONES + [c for c in CONSONANT_PHONES if c not in VOWEL_PHONES]

_TELUGU_BLOCK = re.compile(r"[ఀ-౿]")


def word_to_phones(word: str) -> list[str]:
    phones: list[str] = []
    pending_inherent = False  # last consonant still owns its inherent 'a'
    for ch in word:
        if ch in CONSONANTS:
            if pending_inherent:
                phones.append("a")
            phones.append(CONSONANTS[ch])
            pending_inherent = True
        elif ch in MATRAS:
            phones.extend(MATRAS[ch])
            pending_inherent = False
        elif ch == AI_LENGTH_MARK:  # e + ai-length-mark sequence == ai
            if phones and phones[-1] == "e":
                phones[-1] = "ai"
        elif ch == VIRAMA:
            pending_inherent = False
        elif ch in INDEPENDENT_VOWELS:
            if pending_inherent:
                phones.append("a")
            phones.extend(INDEPENDENT_VOWELS[ch])
            pending_inherent = False
        elif ch in (ANUSVARA, CANDRABINDU):
            if pending_inherent:
                phones.append("a")
            phones.append(_NASAL_PLACEHOLDER)
            pending_inherent = False
        elif ch == VISARGA:
            if pending_inherent:
                phones.append("a")
            phones.append("h")
            pending_inherent = False
        # NUKTA, LENGTH_MARK and anything else are ignored
    if pending_inherent:
        phones.append("a")

    resolved = []
    for i, p in enumerate(phones):
        if p == _NASAL_PLACEHOLDER:
            nxt = phones[i + 1] if i + 1 < len(phones) else None
            resolved.append(_HOMORGANIC.get(nxt, "m"))
        else:
            resolved.append(p)
    return resolved


class TeluguFrontend(TextFrontend):
    language = "telugu"

    @property
    def symbols(self) -> list[str]:
        return list(TELUGU_PHONES)

    def normalize(self, text: str) -> str:
        return normalize_telugu(text)

    def is_valid_text(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        telugu = len(_TELUGU_BLOCK.findall(text))
        return telugu / max(1, sum(not ch.isspace() for ch in text)) > 0.7

    def _g2p(self, normalized: str) -> list[WordPron]:
        out: list[WordPron] = []
        # split into words and punctuation, keeping punctuation as separate tokens
        for token in re.findall(r"[ఀ-౿]+|[^\sఀ-౿]", normalized):
            if _TELUGU_BLOCK.match(token):
                phones = word_to_phones(token)
                if phones:
                    out.append(WordPron(word=token, phones=phones))
            elif token in LONG_PAUSE_PUNCT:
                out.append(WordPron(word=token, phones=[SIL]))
            elif token in SHORT_PAUSE_PUNCT:
                out.append(WordPron(word=token, phones=[SP]))
        return out
