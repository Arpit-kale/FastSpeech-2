"""Train FastSpeech 2 (resumes automatically from checkpoints/last.pt) and plot losses.

    python train.py
    python train.py --set train.total_steps=60000 --set train.max_frames_per_batch=8000
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fastspeech2.train import Trainer  # noqa: E402
from fastspeech2.utils import load_config  # noqa: E402
from fastspeech2.utils.plots import plot_losses  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--plot-only", action="store_true", help="only regenerate loss plots")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if not args.plot_only:
        Trainer(cfg).fit()
    plot_losses(cfg.paths.log_dir, cfg.paths.results_dir, cfg.language)


if __name__ == "__main__":
    main()
