"""Download external assets.

    python scripts/download_assets.py --vocoder        # universal HiFi-GAN -> pretrained/
    python scripts/download_assets.py --nltk           # g2p_en's NLTK resources
    python scripts/download_assets.py --checkpoints    # trained models from URLs in env vars

Checkpoint URLs (e.g. a Hugging Face model repo or a GitHub release asset):
    FS2_ENGLISH_CHECKPOINT_URL, FS2_TELUGU_CHECKPOINT_URL
They are saved as <Language>/checkpoints/model_inference.pt when missing.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def vocoder() -> None:
    from fastspeech2.utils.model import download_universal_hifigan

    path = download_universal_hifigan(ROOT / "pretrained" / "hifigan_universal.pth.tar")
    print(f"vocoder: {path} ({path.stat().st_size / 2**20:.1f} MB)")


def nltk_data() -> None:
    from fastspeech2.text.english import ensure_nltk_data

    ensure_nltk_data()
    print("nltk: ok")


def checkpoints() -> None:
    for lang, folder in (("ENGLISH", "English"), ("TELUGU", "Telugu")):
        url = os.environ.get(f"FS2_{lang}_CHECKPOINT_URL")
        dest = ROOT / folder / "checkpoints" / "model_inference.pt"
        if not url or dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {folder} checkpoint from {url}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocoder", action="store_true")
    ap.add_argument("--nltk", action="store_true")
    ap.add_argument("--checkpoints", action="store_true")
    a = ap.parse_args()
    if not (a.vocoder or a.nltk or a.checkpoints):
        a.vocoder = a.nltk = True
    if a.vocoder:
        vocoder()
    if a.nltk:
        nltk_data()
    if a.checkpoints:
        checkpoints()


if __name__ == "__main__":
    main()
