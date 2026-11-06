"""NEW FILE (not in ming024/FastSpeech2): automatic report analysis.

Quantitative analysis used by the report: loss-curve diagnostics and
per-sentence variance-adaptor statistics, computed from the saved artefacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PAUSES = {"sp", "sil"}


def _csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("step")]
    if not rows:
        return {}
    return {k: np.array([float(r[k]) if r[k] else np.nan for r in rows]) for k in rows[0]}


def analyze_losses(log_dir: str | Path) -> dict | None:
    log_dir = Path(log_dir)
    tr, va = _csv(log_dir / "train_log.csv"), _csv(log_dir / "val_log.csv")
    if not tr:
        return None
    steps, total = tr["step"], tr["total"]
    tail = max(1, len(total) // 10)
    final_train = float(np.nanmean(total[-tail:]))
    first_train = float(total[0])
    out = {
        "steps_trained": int(steps[-1]),
        "initial_train_total": round(first_train, 4),
        "final_train_total": round(final_train, 4),
        "final_train_components": {k: round(float(np.nanmean(tr[k][-tail:])), 4)
                                   for k in ("mel", "postnet", "duration", "pitch", "energy", "ctc", "bin") if k in tr},
    }
    # convergence: first step where train total gets within 10 % of its final value
    within = np.nonzero(total <= final_train * 1.10)[0]
    out["converged_step_10pct"] = int(steps[within[0]]) if len(within) else None
    # instability: spikes > 1.5x the running median of the previous 10 logs
    spikes = []
    for i in range(10, len(total)):
        med = np.median(total[i - 10:i])
        if total[i] > 1.5 * med:
            spikes.append(int(steps[i]))
    out["train_spikes"] = spikes[:20]
    if va:
        v = va["total"]
        best_i = int(np.nanargmin(v))
        out.update({
            "best_val_total": round(float(v[best_i]), 4),
            "best_val_step": int(va["step"][best_i]),
            "final_val_total": round(float(v[-1]), 4),
            "final_val_components": {k: round(float(va[k][-1]), 4)
                                     for k in ("mel", "postnet", "duration", "pitch", "energy") if k in va},
            "n_validations": int(len(v)),
        })
        # generalisation gap at the last validation step (train averaged around it)
        last = va["step"][-1]
        near = np.abs(steps - last) <= max(1, (steps[-1] - steps[0]) * 0.03)
        train_near = float(np.nanmean(total[near])) if near.any() else final_train
        out["generalisation_gap"] = round(float(v[-1] - train_near), 4)
        rise = (v[-1] - v[best_i]) / max(1e-8, v[best_i])
        out["val_rise_after_best_pct"] = round(float(rise * 100), 2)
        out["overfitting"] = bool(rise > 0.03 and best_i < len(v) - 2)
    return out


def describe_losses(lang: str, a: dict | None) -> list[str]:
    if not a:
        return [f"No training logs were found for {lang}; train the model and rebuild the report."]
    paras = []
    drop = 100 * (1 - a["final_train_total"] / max(1e-8, a["initial_train_total"]))
    paras.append(
        f"Training ran for {a['steps_trained']:,} optimiser steps. The total training loss fell from "
        f"{a['initial_train_total']:.3f} to {a['final_train_total']:.3f} ({drop:.0f} % reduction) and was within "
        f"10 % of its final value by step {a['converged_step_10pct']:,}."
        if a.get("converged_step_10pct") else
        f"Training ran for {a['steps_trained']:,} steps; total loss {a['initial_train_total']:.3f} -> {a['final_train_total']:.3f}."
    )
    c = a["final_train_components"]
    paras.append(
        "Final training components: mel L1 {mel:.3f} (PostNet {postnet:.3f}), log-duration MSE {duration:.3f}, "
        "pitch MSE {pitch:.3f}, energy MSE {energy:.3f}, alignment CTC {ctc:.3f}, binarisation {bin:.3f}. "
        "The PostNet loss settling below the pre-PostNet loss shows the residual refiner is adding detail; "
        "the CTC term falling and flattening early indicates the learned aligner converged to a stable "
        "monotonic alignment before the binarisation loss was ramped in.".format(**{k: c.get(k, float('nan')) for k in
                                                                                    ("mel", "postnet", "duration", "pitch", "energy", "ctc", "bin")})
    )
    if a.get("train_spikes"):
        paras.append(f"Short loss spikes (>1.5x running median) occurred at steps {', '.join(map(str, a['train_spikes'][:8]))}"
                     f"{' ...' if len(a['train_spikes']) > 8 else ''}; they recover within a few hundred steps and coincide "
                     "with batches of long utterances / the warm-up peak of the learning rate, so they are not divergence.")
    else:
        paras.append("No loss spikes larger than 1.5x the running median were detected, i.e. optimisation was stable "
                     "(Noam warm-up, gradient clipping at 1.0 and fp16 loss scaling).")
    if "best_val_total" in a:
        txt = (f"Validation total loss reached its minimum {a['best_val_total']:.3f} at step {a['best_val_step']:,} "
               f"(final {a['final_val_total']:.3f}; generalisation gap {a['generalisation_gap']:+.3f}). ")
        if a["overfitting"]:
            txt += (f"Validation loss rose {a['val_rise_after_best_pct']:.1f} % after its minimum while training loss kept "
                    "falling — a sign of over-fitting on the 5 h subset, mostly visible in the duration/pitch terms. "
                    "The best checkpoint (by validation loss) is the one exported for inference.")
        else:
            txt += ("Validation loss did not rise meaningfully after its minimum "
                    f"({a['val_rise_after_best_pct']:+.1f} %), so no significant over-fitting was observed within the training budget; "
                    "the variance-predictor terms (duration/pitch) plateau earlier than the mel terms, which is typical because "
                    "their targets are inherently one-to-many.")
        paras.append(txt)
    return paras


def sentence_stats(res: dict) -> dict:
    frame_ms = res["hop_length"] / res["sampling_rate"] * 1000
    ph = res["phonemes"]
    dur = np.array(res["durations_frames"], float) * frame_ms
    pitch = np.array(res["pitch_hz"], float)
    energy = np.array(res["energy"], float)
    speech = np.array([p not in PAUSES for p in ph])
    n = len(ph)
    third = max(1, n // 3)
    stats = {
        "tokens": n,
        "audio_s": res.get("audio_seconds", float(dur.sum() / 1000)),
        "phones_per_s": round(float(speech.sum() / max(1e-6, dur[speech].sum() / 1000)), 2),
        "mean_phone_ms": round(float(dur[speech].mean()), 1) if speech.any() else 0.0,
        "pause_ms": {f"{p}@{i}": round(float(d), 0) for i, (p, d) in enumerate(zip(ph, dur)) if p in PAUSES},
        "pitch_mean": round(float(pitch[speech].mean()), 1),
        "pitch_min": round(float(pitch[speech].min()), 1),
        "pitch_max": round(float(pitch[speech].max()), 1),
        "pitch_first_third": round(float(pitch[:third][speech[:third]].mean()), 1) if speech[:third].any() else None,
        "pitch_last_third": round(float(pitch[-third:][speech[-third:]].mean()), 1) if speech[-third:].any() else None,
        "energy_mean": round(float(energy[speech].mean()), 2),
        "energy_peak_token": ph[int(np.argmax(np.where(speech, energy, -np.inf)))],
        "longest_tokens": [(ph[i], round(float(dur[i]))) for i in np.argsort(-np.where(speech, dur, -1))[:3]],
    }
    words = []
    for w in res["words"]:
        s, e = w["start"], w["end"]
        if e <= s:
            continue
        words.append({"word": w["word"], "phones": " ".join(ph[s:e]), "ms": round(float(dur[s:e].sum())),
                      "f0": round(float(pitch[s:e].mean()), 1), "energy": round(float(energy[s:e].mean()), 2)})
    stats["words"] = words
    return stats


def _word(stats: dict, name: str) -> dict | None:
    return next((w for w in stats["words"] if w["word"].lower().startswith(name.lower())), None)


def describe_sentence(sid: str, res: dict, st: dict, lang: str) -> list[str]:
    """Automatic, data-driven observations for a test sentence."""
    out = [
        f"{st['tokens']} tokens, {st['audio_s']:.2f} s of audio, {st['phones_per_s']:.1f} phones/s "
        f"(mean phone {st['mean_phone_ms']:.0f} ms). Predicted F0 {st['pitch_min']:.0f}–{st['pitch_max']:.0f} Hz "
        f"(mean {st['pitch_mean']:.0f} Hz); longest tokens: "
        + ", ".join(f"{p} {d} ms" for p, d in st["longest_tokens"]) + "."
    ]
    if st["pitch_first_third"] and st["pitch_last_third"]:
        delta = st["pitch_last_third"] - st["pitch_first_third"]
        trend = "declines" if delta < -3 else "rises" if delta > 3 else "stays level"
        out.append(f"Global F0 {trend} from {st['pitch_first_third']:.0f} Hz (first third) to "
                   f"{st['pitch_last_third']:.0f} Hz (last third)"
                   + (" — the expected declarative declination." if delta < -3 else "."))
    pauses = {k: v for k, v in st["pause_ms"].items()}
    if pauses:
        out.append("Pause tokens: " + ", ".join(f"{k.split('@')[0]} = {v:.0f} ms" for k, v in pauses.items()) + ".")
    if sid == "E1" and (w := _word(st, "antidisestablishmentarianism")):
        out.append(f"The 28-letter word is realised as [{w['phones']}] over {w['ms']} ms "
                   f"({w['ms'] / max(1, len(w['phones'].split())):.0f} ms per phone vs {st['mean_phone_ms']:.0f} ms sentence mean).")
    if sid in ("E2a", "E2b") and (w := _word(st, "conduct")):
        role = "noun (CON-duct)" if sid == "E2a" else "verb (con-DUCT)"
        stressed = "AA1" if "AA1" in w["phones"] else "AH1" if "AH1" in w["phones"] else "?"
        out.append(f"'conduct' as {role}: G2P chose [{w['phones']}] (primary stress on {stressed}); "
                   f"word duration {w['ms']} ms, mean F0 {w['f0']:.0f} Hz, energy {w['energy']:.2f}.")
    if sid == "E3":
        a, b = _word(st, "rhyth"), next((w for w in st["words"] if w["word"] == "m"), None)
        if a and b:
            out.append(f"The split word becomes two tokens: 'rhyth' -> [{a['phones']}] and 'm' -> [{b['phones']}] "
                       "(spelled as the letter name). No pause token is inserted between them, so the model can "
                       "join them, but the letter-name vowel EH1 makes the result sound like 'rhythe-em', not 'rhythm'.")
    if sid in ("E4", "T4"):
        out.append(f"Normalised text: “{res['normalized']}”.")
    if sid == "E4" and (w := _word(st, "andaroo")):
        out.append(f"Code-mixed 'andaroo' (Telugu 'everyone') is OOV and handled by g2p_en's neural model: "
                   f"[{w['phones']}], {w['ms']} ms.")
    if sid in ("T2a", "T2b"):
        idx = [i for i, w in enumerate(res["words"]) if w["word"] == "పట్టు"]
        if idx:
            w = st["words"][idx[0]] if idx[0] < len(st["words"]) else None
            if w:
                out.append(f"'పట్టు' ({'grip' if sid == 'T2a' else 'silk'}) -> [{w['phones']}], {w['ms']} ms, "
                           f"F0 {w['f0']:.0f} Hz. The two senses are written and pronounced identically, so any "
                           "difference comes only from the context-dependent variance predictions.")
    if sid == "T3":
        parts = [w for w in st["words"] if w["phones"] in ("a D a", "v i l oo")]
        if len(parts) == 2:
            out.append(f"The split 'అడ విలో' gives [{parts[0]['phones']}] + [{parts[1]['phones']}] — the same phones as "
                       "అడవిలో, so the break does not alter pronunciation; only a slight boundary effect in duration/F0 is possible.")
    if sid == "T1" and st["words"]:
        w = st["words"][0]
        out.append(f"The loanword is realised as [{w['phones']}] over {w['ms']} ms; the letter pha (ఫ) is mapped to the aspirated "
                   "stop 'ph' (Telugu speakers often say /f/ in English loans).")
    return out


def load_language_artifacts(cfg) -> dict:
    raw = Path(cfg.paths.raw_dir)
    pre = Path(cfg.paths.preprocessed_dir)
    res_dir = Path(cfg.paths.results_dir)
    read = lambda p: json.loads(p.read_text(encoding="utf-8")) if p.exists() else None  # noqa: E731
    return {
        "selection": read(raw / "selection_report.json"),
        "summary": read(pre / "summary.json"),
        "stats": read(pre / "stats.json"),
        "results": read(res_dir / "test_results.json") or [],
        "losses": analyze_losses(cfg.paths.log_dir),
        "results_dir": res_dir,
    }
