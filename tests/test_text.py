"""Text frontend tests (fast, no audio)."""

import pytest

from fastspeech2.text import Vocabulary, get_frontend
from fastspeech2.text.numbers import cardinal, ordinal, year
from fastspeech2.text.numbers_te import cardinal_te, ordinal_te
from fastspeech2.text.telugu import word_to_phones


def test_english_numbers():
    assert cardinal(45) == "forty five"
    assert ordinal(12) == "twelfth"
    assert ordinal(21) == "twenty first"
    assert year(2026) == "twenty twenty six"
    assert year(2005) == "two thousand five"


def test_english_normalization_e4():
    fe = get_frontend("english")
    norm = fe.normalize("The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.")
    assert "ten forty five ay em" in norm
    assert "the twelfth of November twenty twenty six" in norm


def test_telugu_numbers():
    assert cardinal_te(2026) == "రెండు వేల ఇరవై ఆరు"
    assert cardinal_te(150) == "నూట యాభై"
    assert cardinal_te(200) == "రెండు వందలు"
    assert ordinal_te(12) == "పన్నెండవ"


def test_telugu_g2p_rules():
    assert word_to_phones("పట్టు") == ["p", "a", "T", "T", "u"]  # gemination
    assert word_to_phones("సమావేశం") == ["s", "a", "m", "aa", "v", "ee", "sh", "a", "m"]  # final anusvara
    assert word_to_phones("బలంగా") == ["b", "a", "l", "a", "ng", "g", "aa"]  # homorganic nasal
    assert word_to_phones("నవంబర్") == ["n", "a", "v", "a", "m", "b", "a", "r"]  # final virama


def test_telugu_t4_normalisation():
    fe = get_frontend("telugu")
    norm = fe.normalize("ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు")
    assert "రెండు వేల ఇరవై ఆరు" in norm and "పన్నెండవ" in norm and "పది నలభై ఐదు" in norm


@pytest.mark.parametrize("lang", ["english", "telugu"])
def test_all_phones_in_vocab_and_single_final_sil(lang):
    import json
    from pathlib import Path

    fe = get_frontend(lang)
    vocab = Vocabulary.from_frontend(fe)
    sentences = json.loads((Path(__file__).parents[1] / "data" / "test_sentences.json").read_text(encoding="utf-8"))
    for item in sentences[lang]:
        phones = fe.phonemize(item["text"])
        assert all(p in vocab for p in phones), item["id"]
        assert phones[-1] == "sil" and phones.count("sil") == 1


def test_english_homograph_conduct():
    fe = get_frontend("english")
    noun = {w.word: w.phones for w in fe.analyze("The orchestra's conduct during the ceremony was praised.").words}
    verb = {w.word: w.phones for w in fe.analyze("The researchers will conduct an experiment.").words}
    assert noun["conduct"][1] == "AA1"  # CON-duct
    assert "AH1" in verb["conduct"]  # con-DUCT
