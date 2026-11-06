"""Modified from ming024/FastSpeech2 ``text/cleaners.py``.

The reference's ``english_cleaners`` (lowercase, expand numbers/abbreviations,
collapse whitespace) are replaced by language-specific normalisers:

* ``english_cleaners`` -> ``text/numbers.py``: times (10:45 AM), dates
  (12 November 2026), years, ordinals, decimals, currency, abbreviations.
* ``telugu_cleaners`` (new) -> ``text/numbers_te.py``: Telugu/ASCII digits,
  Indian-system cardinals, ordinals (12వ), times, Latin acronyms.
"""

from .numbers import normalize_english
from .numbers_te import normalize_telugu


def english_cleaners(text: str) -> str:
    return normalize_english(text)


def telugu_cleaners(text: str) -> str:
    return normalize_telugu(text)
