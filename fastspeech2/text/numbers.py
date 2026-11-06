"""Modified from ming024/FastSpeech2 ``text/numbers.py`` (adds times, dates, years, currency, ordinals).

English text normalisation: numbers, ordinals, years, times, dates, symbols."""

from __future__ import annotations

import re
import unicodedata

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINAL_IRREGULAR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}
_SCALES = ((10**9, "billion"), (10**6, "million"), (1000, "thousand"))
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]
_MONTH_RE = "|".join(MONTHS + [m[:3] for m in MONTHS])
_ABBREVIATIONS = {
    "mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor", "prof": "professor",
    "st": "saint", "jr": "junior", "sr": "senior", "vs": "versus", "etc": "et cetera",
    "no": "number", "govt": "government", "dept": "department",
}


def cardinal(n: int) -> str:
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + ("" if rest == 0 else " " + _ONES[rest])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + ("" if rest == 0 else " and " + cardinal(rest))
    for value, name in _SCALES:
        if n >= value:
            q, rest = divmod(n, value)
            joiner = " and " if 0 < rest < 100 else " "
            return cardinal(q) + " " + name + ("" if rest == 0 else joiner + cardinal(rest))
    raise AssertionError("unreachable")


def ordinal(n: int) -> str:
    words = cardinal(n).split()
    last = words[-1]
    if last in _ORDINAL_IRREGULAR:
        last = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    words[-1] = last
    return " ".join(words)


def year(n: int) -> str:
    """2026 -> 'twenty twenty six', 2005 -> 'two thousand five', 1900 -> 'nineteen hundred'."""
    if 2001 <= n <= 2009:
        return "two thousand " + _ONES[n - 2000]
    if not 1000 <= n <= 9999 or n % 1000 == 0:
        return cardinal(n)
    first, last = divmod(n, 100)
    if last == 0:
        return cardinal(first) + " hundred"
    if last < 10:
        return cardinal(first) + " oh " + _ONES[last]
    return cardinal(first) + " " + cardinal(last)


def _time(hours: int, minutes: int, ampm: str | None) -> str:
    h = cardinal(hours)
    if minutes == 0:
        m = "" if ampm else " o'clock"
    elif minutes < 10:
        m = " oh " + _ONES[minutes]
    else:
        m = " " + cardinal(minutes)
    suffix = ""
    if ampm:
        suffix = " ay em" if ampm.lower().startswith("a") else " pee em"
    return f"{h}{m}{suffix}"


def _month_full(token: str) -> str:
    for m in MONTHS:
        if token.lower() == m.lower() or token.lower() == m[:3].lower():
            return m
    return token


def normalize_english(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

    # Times: 10:45 AM, 7:05pm, 18:30
    t = re.sub(
        r"\b(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]\.?(?=\W|$)",
        lambda m: _time(int(m.group(1)), int(m.group(2)), m.group(3)),
        t,
    )
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: _time(int(m.group(1)), int(m.group(2)), None), t)
    t = re.sub(
        r"\b(\d{1,2})\s*([AaPp])\.?[Mm]\.?(?=\W|$)",
        lambda m: _time(int(m.group(1)), 0, m.group(2)),
        t,
    )

    # Dates: "12 November 2026", "12th Nov, 2026", "November 12, 2026"
    t = re.sub(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?,?\s+(\d{{4}})\b",
        lambda m: f"the {ordinal(int(m.group(1)))} of {_month_full(m.group(2))} {year(int(m.group(3)))}",
        t,
    )
    t = re.sub(
        rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
        lambda m: f"{_month_full(m.group(1))} {ordinal(int(m.group(2)))}, {year(int(m.group(3)))}",
        t,
    )
    t = re.sub(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\b",
        lambda m: f"the {ordinal(int(m.group(1)))} of {_month_full(m.group(2))}",
        t,
    )

    # Currency and percentages
    t = re.sub(r"[$]\s?(\d[\d,]*)", lambda m: cardinal(int(m.group(1).replace(",", ""))) + " dollars", t)
    t = re.sub(r"(?:₹|Rs\.?)\s?(\d[\d,]*)", lambda m: cardinal(int(m.group(1).replace(",", ""))) + " rupees", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s?%", r"\1 percent", t)

    # Ordinals (21st), decimals (3.5), years, plain integers
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), t)
    t = re.sub(
        r"\b(\d+)\.(\d+)\b",
        lambda m: cardinal(int(m.group(1))) + " point " + " ".join(_ONES[int(d)] for d in m.group(2)),
        t,
    )
    t = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: year(int(m.group(1))), t)
    t = re.sub(r"\b\d[\d,]*\b", lambda m: cardinal(int(m.group(0).replace(",", ""))), t)

    # Abbreviations and symbols
    t = re.sub(
        r"\b([A-Za-z]+)\.(?=\s)",
        lambda m: _ABBREVIATIONS.get(m.group(1).lower(), m.group(0)),
        t,
    )
    t = t.replace("&", " and ").replace("+", " plus ").replace("@", " at ")
    t = re.sub(r"\s+", " ", t).strip()
    return t
