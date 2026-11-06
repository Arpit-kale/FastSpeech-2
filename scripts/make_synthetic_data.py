"""Create a tiny synthetic dataset (real text, vocoder-free synthetic audio) for smoke tests / CI.

Each phone becomes a short segment: vowels/sonorants are harmonic tones with a
phone-dependent F0 and formant-like spectrum, other consonants are filtered
noise. This exercises every stage (G2P, features, aligner, training,
inference, plots) in minutes on a CPU without downloading any dataset.

    python scripts/make_synthetic_data.py --language english --out data/smoke/english
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastspeech2.text import get_frontend  # noqa: E402

WORDS = {
    "english": ("the quick brown fox jumps over a lazy dog while seven bright students read "
                "short stories about history culture music and science in the morning").split(),
    "telugu": ("ఈ రోజు మేము పాఠశాలలో తెలుగు కథలు చదివాము పిల్లలు ఆనందంగా పాటలు పాడారు "
               "అమ్మ వంట చేసింది నాన్న పుస్తకం చదివాడు").split(),
}
VOICED = set("aeiouAEIOU") | {"m", "n", "l", "r", "y", "v", "w", "N", "L", "M", "R", "W", "Y"}


def synth_phone(phone: str, sr: int, rng: np.random.Generator) -> np.ndarray:
    dur = rng.uniform(0.06, 0.13)
    t = np.arange(int(dur * sr)) / sr
    h = sum(ord(c) for c in phone)
    if phone in ("sp", "sil"):
        return np.zeros(int(0.12 * sr), dtype=np.float32)
    if phone[0] in VOICED:
        f0 = 110 + (h % 9) * 12 + 15 * np.sin(2 * np.pi * 3 * t)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        y = sum((0.6 / k) * np.sin(k * phase) * (1 + 0.8 * np.cos(k * (h % 5))) for k in range(1, 12))
    else:
        y = rng.normal(0, 0.3, len(t))
        k = 3 + h % 12
        y = np.convolve(y, np.ones(k) / k, mode="same")
    env = np.minimum(1, np.minimum(t, t[-1] - t + 1e-4) * 80)
    return (y * env).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["english", "telugu"], default="english")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sr", type=int, default=22050)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    pyrng = random.Random(0)
    fe = get_frontend(args.language)
    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for i in range(args.n):
            text = " ".join(pyrng.sample(WORDS[args.language], pyrng.randint(4, 8))) + "."
            phones = fe.phonemize(text)
            wav = np.concatenate([synth_phone(p, args.sr, rng) for p in phones])
            wav = 0.9 * wav / (np.abs(wav).max() + 1e-6)
            uid = f"syn_{i:04d}"
            sf.write(out / "wavs" / f"{uid}.wav", wav, args.sr)
            fh.write(json.dumps({"id": uid, "speaker": "spk_a" if i % 3 else "spk_b", "text": text,
                                 "wav": f"wavs/{uid}.wav", "duration": len(wav) / args.sr,
                                 "split": "val" if i % 8 == 0 else "train"}, ensure_ascii=False) + "\n")
    print(f"wrote {args.n} utterances to {out}")


if __name__ == "__main__":
    main()
