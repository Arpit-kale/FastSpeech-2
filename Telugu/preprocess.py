"""Download the dataset, select ~5 h train / ~1 h val, and extract features.

    python preprocess.py                 # full stage 1 + 2
    python preprocess.py --skip-download # only re-run feature extraction
    python preprocess.py --manifest my_manifest.jsonl
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fastspeech2.preprocessor.hf_subset import prepare  # noqa: E402
from fastspeech2.preprocessor.preprocessor import preprocess  # noqa: E402
from fastspeech2.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--manifest", help="use an existing manifest.jsonl instead of downloading")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    manifest = args.manifest
    if not manifest and not args.skip_download:
        manifest = prepare(cfg)
    preprocess(cfg, manifest, args.workers)


if __name__ == "__main__":
    main()
