"""Modified from ming024/FastSpeech2 ``preprocessor/preprocessor.py`` (no TextGrids; per-speaker pitch stats; multiprocessing).

Stage 2 — phonemise text and extract mel / pitch / energy features.

Outputs (under ``paths.preprocessed_dir``)::

    mel/<id>.npy        log-mel  [T, 80]
    pitch/<id>.npy      speaker-normalised, interpolated F0 per frame [T]
    energy/<id>.npy     globally normalised frame energy [T]
    train.txt, val.txt  id|speaker|phones|raw text
    speakers.json       speaker -> index
    stats.json          pitch (per speaker, Hz) and energy statistics + bin ranges
    vocab.json          phone inventory (ids are stored in checkpoints too)

Unlike ming024 there are no MFA TextGrids: durations, and therefore the
phoneme-level pitch/energy targets, are produced on the fly by the learned
aligner during training (see ``model/aligner.py``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..audio import AudioProcessor
from ..text import Vocabulary, get_frontend
from ..utils import ensure_dir, get_logger

log = get_logger("preprocess")

_WORKER: dict = {}


def _init_worker(audio_cfg: dict, out_dir: str) -> None:
    _WORKER["proc"] = AudioProcessor(audio_cfg)
    _WORKER["out"] = Path(out_dir)


def _extract(item: dict) -> dict | None:
    proc: AudioProcessor = _WORKER["proc"]
    out: Path = _WORKER["out"]
    try:
        wav = proc.load(item["wav_abs"], trim=False)  # already trimmed in stage 1
        mel, energy = proc.mel_and_energy(wav)
        n = mel.shape[1]
        f0 = proc.pitch(wav, n)
        voiced = f0 > 0
        if voiced.sum() < 5:
            return None
        np.save(out / "mel" / f"{item['id']}.npy", mel.T.astype(np.float32))
        np.save(out / "raw_f0" / f"{item['id']}.npy", f0)
        np.save(out / "energy" / f"{item['id']}.npy", energy)
        return {**item, "n_frames": int(n), "f0_voiced": f0[voiced].astype(np.float64)}
    except Exception as exc:  # pragma: no cover - logged and skipped
        return {"error": f"{item['id']}: {exc}"}


def preprocess(cfg, manifest_path: str | Path | None = None, num_workers: int | None = None) -> Path:
    raw_dir = Path(cfg.paths.raw_dir)
    manifest_path = Path(manifest_path or raw_dir / "manifest.jsonl")
    out = ensure_dir(cfg.paths.preprocessed_dir)
    for sub in ("mel", "pitch", "energy", "raw_f0"):
        ensure_dir(out / sub)

    frontend = get_frontend(cfg.language)
    vocab = Vocabulary.from_frontend(frontend)
    items, skipped_text = [], 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        phones = frontend.phonemize(rec["text"])
        if len(phones) < 2:
            skipped_text += 1
            continue
        wav_abs = Path(rec["wav"])
        if not wav_abs.is_absolute():
            wav_abs = manifest_path.parent / wav_abs
        items.append({**rec, "wav_abs": str(wav_abs), "phones": phones})
    log.info("%d utterances to process (%d skipped by text frontend)", len(items), skipped_text)

    workers = num_workers or int(cfg.preprocessing.get("num_workers", 0)) or max(1, (__import__("os").cpu_count() or 2) - 1)
    results = []
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(dict(cfg.audio), str(out))) as ex:
            for r in tqdm(ex.map(_extract, items, chunksize=8), total=len(items), desc="features"):
                results.append(r)
    else:
        _init_worker(dict(cfg.audio), str(out))
        results = [_extract(it) for it in tqdm(items, desc="features")]

    ok = []
    for r in results:
        if r is None:
            continue
        if "error" in r:
            log.warning("skip %s", r["error"])
            continue
        # The learned aligner needs at least one frame per token.
        if r["n_frames"] < len(r["phones"]) + 2:
            continue
        ok.append(r)
    log.info("%d utterances kept after feature extraction", len(ok))
    if not ok:
        raise RuntimeError("No utterances survived preprocessing.")

    # ---- speaker-wise pitch statistics (Hz, voiced frames only)
    voiced_by_spk = defaultdict(list)
    for r in ok:
        voiced_by_spk[r["speaker"]].append(r["f0_voiced"])
    pitch_stats = {}
    for spk, chunks in voiced_by_spk.items():
        v = np.concatenate(chunks)
        lo, hi = np.percentile(v, [1, 99])  # robust to octave errors
        v = v[(v >= lo) & (v <= hi)]
        pitch_stats[spk] = [float(v.mean()), float(max(v.std(), 1.0))]

    # ---- normalise pitch (per speaker) and energy (global)
    all_energy = np.concatenate([np.load(out / "energy" / f"{r['id']}.npy") for r in ok])
    e_mean, e_std = float(all_energy.mean()), float(max(all_energy.std(), 1e-5))
    p_min = p_max = e_min = e_max = None
    for r in tqdm(ok, desc="normalise"):
        f0 = np.load(out / "raw_f0" / f"{r['id']}.npy")
        mean, std = pitch_stats[r["speaker"]]
        f0 = AudioProcessor.interpolate_f0(f0)
        f0 = np.clip(f0, mean - 4 * std, mean + 4 * std)
        p = ((f0 - mean) / std).astype(np.float32)
        e = ((np.load(out / "energy" / f"{r['id']}.npy") - e_mean) / e_std).astype(np.float32)
        np.save(out / "pitch" / f"{r['id']}.npy", p)
        np.save(out / "energy" / f"{r['id']}.npy", e)
        p_min = p.min() if p_min is None else min(p_min, p.min())
        p_max = p.max() if p_max is None else max(p_max, p.max())
        e_min = e.min() if e_min is None else min(e_min, e.min())
        e_max = e.max() if e_max is None else max(e_max, e.max())

    speakers = sorted({r["speaker"] for r in ok})
    speaker_map = {s: i for i, s in enumerate(speakers)}
    spk_frames = defaultdict(int)
    for r in ok:
        if r["split"] == "train":
            spk_frames[r["speaker"]] += r["n_frames"]
    default_speaker = max(spk_frames, key=spk_frames.get) if spk_frames else speakers[0]

    stats = {
        "pitch": {"per_speaker_hz": pitch_stats, "min": float(p_min), "max": float(p_max)},
        "energy": {"mean": e_mean, "std": e_std, "min": float(e_min), "max": float(e_max)},
        "hop_length": int(cfg.audio.hop_length),
        "sampling_rate": int(cfg.audio.sampling_rate),
        "default_speaker": default_speaker,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out / "speakers.json").write_text(json.dumps(speaker_map, indent=2), encoding="utf-8")
    vocab.save(out / "vocab.json")

    summary = defaultdict(lambda: {"utterances": 0, "hours": 0.0, "phones": 0})
    for split in ("train", "val"):
        rows = [r for r in ok if r["split"] == split]
        with open(out / f"{split}.txt", "w", encoding="utf-8") as fh:
            for r in rows:
                text = r["text"].replace("|", " ").replace("\n", " ")
                fh.write(f"{r['id']}|{r['speaker']}|{' '.join(r['phones'])}|{text}\n")
                summary[split]["utterances"] += 1
                summary[split]["hours"] += r["n_frames"] * cfg.audio.hop_length / cfg.audio.sampling_rate / 3600
                summary[split]["phones"] += len(r["phones"])
    summary = {k: {**v, "hours": round(v["hours"], 3)} for k, v in summary.items()}
    (out / "summary.json").write_text(json.dumps({"splits": summary, "speakers": len(speakers),
                                                  "default_speaker": default_speaker}, indent=2),
                                      encoding="utf-8")
    log.info("Preprocessing done: %s", json.dumps(summary))
    return out
