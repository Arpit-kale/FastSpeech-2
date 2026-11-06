"""Data selection test against an in-memory dataset with the IndicTTS Telugu schema."""

import io
import json

import numpy as np
import pytest
import soundfile as sf

datasets = pytest.importorskip("datasets")


def _wav_bytes(seconds: float, sr: int = 48000) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * 150 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def test_prepare_selects_disjoint_single_speaker(tmp_path, monkeypatch):
    from datasets import Audio, ClassLabel, Dataset, Features, Value

    from fastspeech2.preprocessor import hf_subset as prep
    from fastspeech2.utils import load_config

    rng = np.random.default_rng(0)
    rows = {"audio": [], "text": [], "gender": []}
    words = "ఈ రోజు మేము పాఠశాలలో తెలుగు కథలు చదివాము పిల్లలు ఆనందంగా పాటలు పాడారు".split()
    for i in range(60):
        dur = float(rng.uniform(1.5, 4.0)) if i != 5 else 40.0  # one over-long clip must be dropped
        rows["audio"].append({"bytes": _wav_bytes(dur), "path": f"utt_{i}.wav"})
        rows["text"].append(" ".join(rng.choice(words, 4)) + f" {i}." if i != 7 else rows["text"][3] if i > 3 else "x")
        rows["gender"].append(0 if i % 4 else 1)
    raw_audio = {"bytes": Value("binary"), "path": Value("string")}
    feats = Features({"audio": raw_audio, "text": Value("string"), "gender": ClassLabel(names=["female", "male"])})
    # Build with a plain struct, then cast to Audio(decode=False) like the HF parquet files.
    ds = Dataset.from_dict(rows, features=feats).cast_column("audio", Audio(decode=False))
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: ds)

    cfg = load_config("Telugu/config.yaml", [
        f"paths.raw_dir={tmp_path}/raw", "dataset.train_hours=0.02", "dataset.val_hours=0.005",
    ])
    manifest = prep.prepare(cfg)
    recs = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    train = [r for r in recs if r["split"] == "train"]
    val = [r for r in recs if r["split"] == "val"]
    assert train and val
    assert not {r["source_index"] for r in train} & {r["source_index"] for r in val}
    assert not {r["text"] for r in train} & {r["text"] for r in val}
    assert all(r["gender"] == "female" for r in recs)  # single preferred speaker
    assert all(r["original_duration"] <= 15.0 for r in recs)
    report = json.loads((tmp_path / "raw" / "selection_report.json").read_text())
    assert report["single_speaker"] is True
