"""Replaces ming024/FastSpeech2 ``prepare_align.py`` + ``preprocessor/ljspeech.py`` (Hugging Face download + 5 h / 1 h selection).

Stage 1 — download a Hugging Face dataset and build a ~5 h / ~1 h train/val subset.

Selection strategy (identical for both languages, fully deterministic via ``seed``):

1. Read only lightweight metadata (text, speaker, gender, duration) for every row.
2. Filter: duration in [min_duration, max_duration], text passes the language
   frontend's script check, text not empty. If the filtered pool cannot supply
   ``train_hours + val_hours`` the max-duration limit is relaxed in 2 s steps up
   to ``max_duration_ceiling``.
3. De-duplicate on normalised text, so an identical prompt can never appear in
   both train and validation (no text leakage).
4. Speaker policy: ``single_speaker: auto`` keeps the preferred / largest
   speaker when it alone covers the budget (best quality for a 5 h TTS model),
   otherwise all speakers are kept and a speaker embedding is used.
5. Validation is drawn first, round-robin over speakers (stratified), then the
   training set is drawn round-robin from the remaining utterances.
6. Selected clips are decoded, resampled to 22.05 kHz, silence-trimmed,
   peak-normalised and written as 16-bit WAV, with a JSONL manifest.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from ..audio import AudioProcessor
from ..text import get_frontend
from ..utils import ensure_dir, get_logger

log = get_logger("prepare")

_SPEAKER_CANDIDATES = ["speaker_id", "speaker", "spk_id", "client_id", "speaker_name"]
_TEXT_CANDIDATES = ["text", "transcript", "transcription", "sentence", "normalized_text"]
_DEFAULT_SPK_REGEX = r"^([A-Za-z0-9]+)[_-]"


@dataclass
class Candidate:
    index: int
    text: str
    speaker: str
    gender: str
    duration: float


def _resolve_columns(ds, cfg) -> tuple[str, str]:
    from datasets import Audio

    audio_col = cfg.get("audio_column", "auto")
    if audio_col in (None, "auto"):
        audio_col = next((k for k, v in ds.features.items() if isinstance(v, Audio)), None)
        if audio_col is None:
            raise ValueError(f"No Audio column found in {list(ds.features)}")
    text_col = cfg.get("text_column", "auto")
    if text_col in (None, "auto"):
        text_col = next((c for c in _TEXT_CANDIDATES if c in ds.features), None)
        if text_col is None:
            raise ValueError(f"No text column found in {list(ds.features)}")
    return audio_col, text_col


def _label_decoder(ds, column: str | None):
    if not column or column not in ds.features:
        return None
    feat = ds.features[column]
    names = getattr(feat, "names", None)
    if names:
        return lambda v: names[v] if isinstance(v, int) and 0 <= v < len(names) else str(v)
    return lambda v: str(v)


def _speakers_from_paths(paths: list[str], regex: str) -> list[str] | None:
    pat = re.compile(regex)
    ids = []
    for p in paths:
        m = pat.search(Path(p or "").name)
        if not m:
            return None
        ids.append(m.group(1))
    n_unique = len(set(ids))
    if n_unique < 2 or n_unique > len(ids) / 2:  # one id per file => not a speaker id
        return None
    return ids


def collect_candidates(ds, cfg, frontend) -> list[Candidate]:
    from datasets import Audio

    audio_col, text_col = _resolve_columns(ds, cfg)
    log.info("Columns: audio=%s text=%s (all: %s)", audio_col, text_col, list(ds.features))
    ds = ds.cast_column(audio_col, Audio(decode=False))

    spk_col = cfg.get("speaker_column", "auto")
    if spk_col in (None, "auto"):
        spk_col = next((c for c in _SPEAKER_CANDIDATES if c in ds.features), None)
    gender_col = cfg.get("gender_column") if cfg.get("gender_column") in ds.features else None
    if gender_col is None and "gender" in ds.features:
        gender_col = "gender"
    spk_decode, gender_decode = _label_decoder(ds, spk_col), _label_decoder(ds, gender_col)
    dur_col = next((c for c in ("duration", "audio_duration") if c in ds.features), None)

    texts, speakers, genders, durations, paths = [], [], [], [], []
    meta_cols = [c for c in (text_col, spk_col, gender_col, dur_col) if c]
    meta = ds.select_columns(meta_cols)
    for row in tqdm(meta, desc="metadata", total=len(meta)):
        texts.append(str(row[text_col] or "").strip())
        speakers.append(spk_decode(row[spk_col]) if spk_col else None)
        genders.append(gender_decode(row[gender_col]) if gender_col else "unknown")
        durations.append(float(row[dur_col]) if dur_col and row[dur_col] is not None else None)

    need_audio = any(d is None for d in durations) or spk_col is None
    if need_audio:
        proc = AudioProcessor
        for i, row in enumerate(tqdm(ds.select_columns([audio_col]), desc="audio headers", total=len(ds))):
            audio = row[audio_col] or {}
            paths.append(audio.get("path") or "")
            if durations[i] is None and audio.get("bytes"):
                durations[i] = proc.duration_of_bytes(audio["bytes"])

    if spk_col is None:
        regex = cfg.get("speaker_path_regex") or _DEFAULT_SPK_REGEX
        derived = _speakers_from_paths(paths, regex) if paths else None
        if derived:
            log.info("Speaker ids derived from file names (%d speakers).", len(set(derived)))
            speakers = derived
        else:
            log.info("No speaker column: using gender as pseudo-speaker label.")
            speakers = [f"spk_{g}" for g in genders]

    out = []
    for i in range(len(texts)):
        if durations[i] is None:
            continue
        out.append(Candidate(i, texts[i], str(speakers[i]), str(genders[i]), float(durations[i])))
    return out


def select_subsets(cands: list[Candidate], cfg, frontend) -> tuple[list[Candidate], list[Candidate], dict]:
    rng = random.Random(int(cfg.get("seed", 1234)))
    train_s, val_s = float(cfg.train_hours) * 3600, float(cfg.val_hours) * 3600
    need = (train_s + val_s) * 1.03
    min_d = float(cfg.min_duration)
    max_d = float(cfg.max_duration)
    ceiling = float(cfg.get("max_duration_ceiling", max_d))

    valid = [c for c in cands if frontend.is_valid_text(c.text)]
    while True:
        pool = [c for c in valid if min_d <= c.duration <= max_d]
        if sum(c.duration for c in pool) >= need or max_d >= ceiling:
            break
        max_d = min(ceiling, max_d + 2.0)
        log.warning("Not enough audio; relaxing max_duration to %.1fs", max_d)

    # de-duplicate on normalised text
    seen, unique = set(), []
    for c in pool:
        key = re.sub(r"\W+", " ", frontend.normalize(c.text).lower()).strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)
    pool = unique

    by_spk: dict[str, list[Candidate]] = defaultdict(list)
    for c in pool:
        by_spk[c.speaker].append(c)
    hours = {s: sum(c.duration for c in v) / 3600 for s, v in by_spk.items()}

    policy = cfg.get("single_speaker", "auto")
    preferred = cfg.get("preferred_speaker")
    chosen = list(by_spk)
    if policy in (True, "auto"):
        ranked = sorted(by_spk, key=lambda s: -hours[s])
        if preferred:
            ranked = sorted(ranked, key=lambda s: 0 if preferred.lower() in s.lower() else 1)
        for spk in ranked:
            if hours[spk] * 3600 >= need or policy is True:
                chosen = [spk]
                break
    log.info("Speaker policy=%s -> using %d speaker(s): %s", policy, len(chosen),
             chosen if len(chosen) <= 10 else f"{chosen[:10]}...")

    queues = {s: rng.sample(by_spk[s], len(by_spk[s])) for s in chosen}

    def draw(budget: float) -> list[Candidate]:
        picked, total = [], 0.0
        order = sorted(queues)
        while total < budget and any(queues[s] for s in order):
            rng.shuffle(order)
            for s in order:
                if queues[s] and total < budget:
                    c = queues[s].pop()
                    picked.append(c)
                    total += c.duration
        return picked

    val = draw(val_s)
    train = draw(train_s)
    assert not {c.index for c in val} & {c.index for c in train}

    stats = {
        "max_duration_used": max_d,
        "candidates_total": len(cands),
        "candidates_after_filters": len(pool),
        "speakers_available": {s: round(h, 3) for s, h in sorted(hours.items(), key=lambda x: -x[1])[:50]},
        "speakers_used": chosen,
        "single_speaker": len(chosen) == 1,
    }
    return train, val, stats


def prepare(cfg) -> Path:
    """Run the full download + selection + export stage. Returns manifest path."""
    dcfg = cfg.dataset
    raw_dir = ensure_dir(cfg.paths.raw_dir)
    manifest_path = raw_dir / "manifest.jsonl"
    if manifest_path.exists() and not dcfg.get("overwrite", False):
        log.info("Manifest exists (%s); skipping download/selection.", manifest_path)
        return manifest_path

    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    log.info("Loading %s [%s] ...", dcfg.hf_name, dcfg.hf_split)
    ds = load_dataset(dcfg.hf_name, dcfg.get("hf_config"), split=dcfg.hf_split, token=token,
                      cache_dir=cfg.paths.get("hf_cache_dir"))

    frontend = get_frontend(cfg.language)
    cands = collect_candidates(ds, dcfg, frontend)
    train, val, stats = select_subsets(cands, dcfg, frontend)

    from datasets import Audio

    audio_col, _ = _resolve_columns(ds, dcfg)
    ds = ds.cast_column(audio_col, Audio(decode=False))
    proc = AudioProcessor(cfg.audio)
    wav_dir = ensure_dir(raw_dir / "wavs")
    totals = Counter()
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for split, items in (("val", val), ("train", train)):
            for c in tqdm(items, desc=f"export {split}"):
                audio = ds[c.index][audio_col]
                data = audio.get("bytes") or Path(audio["path"]).read_bytes()
                wav = proc.load(data, trim=True)
                if len(wav) < proc.sr * 0.5:
                    continue
                uid = f"{cfg.language[:2]}_{c.index:06d}"
                spk = re.sub(r"[^A-Za-z0-9_]+", "_", c.speaker)
                path = wav_dir / f"{uid}.wav"
                proc.save(wav, path)
                dur = len(wav) / proc.sr
                totals[split] += dur
                fh.write(json.dumps({
                    "id": uid, "speaker": spk, "gender": c.gender, "text": c.text,
                    "wav": str(path.relative_to(raw_dir)), "duration": round(dur, 3),
                    "original_duration": round(c.duration, 3), "source_index": c.index, "split": split,
                }, ensure_ascii=False) + "\n")

    stats.update({
        "dataset": dcfg.hf_name,
        "split_used": dcfg.hf_split,
        "seed": dcfg.get("seed", 1234),
        "train_utterances": len(train),
        "val_utterances": len(val),
        "train_hours_after_trim": round(totals["train"] / 3600, 3),
        "val_hours_after_trim": round(totals["val"] / 3600, 3),
        "train_hours_before_trim": round(sum(c.duration for c in train) / 3600, 3),
        "val_hours_before_trim": round(sum(c.duration for c in val) / 3600, 3),
    })
    (raw_dir / "selection_report.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Selection: %s", json.dumps({k: v for k, v in stats.items() if k != "speakers_available"}))
    return manifest_path


def candidates_to_dicts(cands: list[Candidate]) -> list[dict]:
    return [asdict(c) for c in cands]
