"""NEW FILE (not in ming024/FastSpeech2): Telugu number normalisation.

Telugu text normalisation: Telugu/ASCII digits, cardinals, ordinals (12వ), times, Latin letters."""

from __future__ import annotations

import re
import unicodedata

_UNITS = [
    "సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
    "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
    "పద్దెనిమిది", "పందొమ్మిది",
]
_TENS = {
    2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై",
}
# (value, singular-standalone, plural-standalone, combining/oblique form)
_SCALES = [
    (10**7, "కోటి", "కోట్లు", "కోట్ల"),
    (10**5, "లక్ష", "లక్షలు", "లక్షల"),
    (1000, "వెయ్యి", "వేలు", "వేల"),
]
_TELUGU_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")
_LATIN_LETTER_NAMES = {
    "a": "ఏ", "b": "బీ", "c": "సీ", "d": "డీ", "e": "ఈ", "f": "ఎఫ్", "g": "జీ", "h": "హెచ్",
    "i": "ఐ", "j": "జే", "k": "కే", "l": "ఎల్", "m": "ఎమ్", "n": "ఎన్", "o": "ఓ", "p": "పీ",
    "q": "క్యూ", "r": "ఆర్", "s": "ఎస్", "t": "టీ", "u": "యూ", "v": "వీ", "w": "డబ్ల్యూ",
    "x": "ఎక్స్", "y": "వై", "z": "జెడ్",
}


def _below_hundred(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    tens, rest = divmod(n, 10)
    return _TENS[tens] + ("" if rest == 0 else " " + _UNITS[rest])


def _below_thousand(n: int) -> str:
    if n < 100:
        return _below_hundred(n)
    hundreds, rest = divmod(n, 100)
    if hundreds == 1:
        return "వంద" if rest == 0 else "నూట " + _below_hundred(rest)
    head = _UNITS[hundreds]
    return f"{head} వందలు" if rest == 0 else f"{head} వందల {_below_hundred(rest)}"


def cardinal_te(n: int) -> str:
    """Indian-system Telugu cardinal: 2026 -> 'రెండు వేల ఇరవై ఆరు'."""
    if n < 0:
        return "మైనస్ " + cardinal_te(-n)
    if n < 1000:
        return _below_thousand(n)
    for value, single, plural, oblique in _SCALES:
        if n >= value:
            q, rest = divmod(n, value)
            if q == 1:
                head = single
                return head if rest == 0 else f"{head} {cardinal_te(rest)}"
            head_num = cardinal_te(q)
            return f"{head_num} {plural}" if rest == 0 else f"{head_num} {oblique} {cardinal_te(rest)}"
    raise AssertionError("unreachable")


def ordinal_te(n: int) -> str:
    """12 -> 'పన్నెండవ', 1 -> 'ఒకటవ', 10 -> 'పదవ'."""
    words = cardinal_te(n).split()
    last = words[-1]
    if last.endswith("ు") or last.endswith("ి"):
        last = last[:-1] + "వ"
    elif last.endswith("ై"):
        last = last[:-1] + "య్యవ"
    else:
        last = last + "వ"
    words[-1] = last
    return " ".join(words)


def _time_te(h: int, m: int) -> str:
    if m == 0:
        return cardinal_te(h)
    return f"{cardinal_te(h)} {cardinal_te(m)}"


def normalize_telugu(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = t.replace("‌", "").replace("‍", "").replace(" ", " ")
    t = t.translate(_TELUGU_DIGITS)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: _time_te(int(m.group(1)), int(m.group(2))), t)
    t = re.sub(r"(\d+)\s?వ(?![ా-్])", lambda m: ordinal_te(int(m.group(1))), t)
    t = re.sub(r"(\d+)\.(\d+)", lambda m: cardinal_te(int(m.group(1))) + " పాయింట్ "
               + " ".join(_UNITS[int(d)] for d in m.group(2)), t)
    t = re.sub(r"\d[\d,]*", lambda m: " " + cardinal_te(int(m.group(0).replace(",", ""))) + " ", t)
    t = re.sub(r"%", " శాతం ", t)
    # Latin letters / acronyms that appear in Telugu text are read letter by letter.
    t = re.sub(r"[A-Za-z]+", lambda m: " " + " ".join(_LATIN_LETTER_NAMES[c.lower()] for c in m.group(0)) + " ", t)
    t = re.sub(r"\s+([,.!?;:।])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
