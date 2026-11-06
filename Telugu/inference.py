"""Synthesise the assignment's test sentences (default) or any custom text.

    python inference.py                                   # all test sentences -> Results/<Lang>/
    python inference.py --text "Hello there" --name demo
    python inference.py --text "..." --duration 1.1 --pitch 1.05 --energy 1.0
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fastspeech2.cli import cmd_synthesize  # noqa: E402
from fastspeech2.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--checkpoint", help="defaults to checkpoints/model_inference.pt or best.pt")
    ap.add_argument("--text")
    ap.add_argument("--name", default="custom")
    ap.add_argument("--out")
    ap.add_argument("--speaker")
    ap.add_argument("--duration", type=float, default=1.0, help=">1 = slower")
    ap.add_argument("--pitch", type=float, default=1.0, help="F0 scale factor")
    ap.add_argument("--energy", type=float, default=1.0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cmd_synthesize(load_config(args.config, args.set), args)


if __name__ == "__main__":
    main()
