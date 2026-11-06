"""NEW FILE (not in ming024/FastSpeech2): single entry point automating all stages.

Command-line entry point that automates every stage of the project.

    python -m fastspeech2.cli <command> --config English/config.yaml [--set key=value ...]

Commands: prepare | preprocess | train | synthesize | plot | report | pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .utils import ensure_dir, get_logger, load_config

log = get_logger("cli")
STAGES = ["prepare", "preprocess", "train", "plot", "synthesize", "report"]


def _cfg(args):
    return load_config(args.config, args.set)


def cmd_prepare(cfg, args=None):
    from .preprocessor.hf_subset import prepare

    return prepare(cfg)


def cmd_preprocess(cfg, args=None):
    from .preprocessor.preprocessor import preprocess

    manifest = getattr(args, "manifest", None) if args else None
    return preprocess(cfg, manifest)


def cmd_train(cfg, args=None):
    from .train import Trainer

    return Trainer(cfg).fit()


def cmd_plot(cfg, args=None):
    from .utils.plots import plot_losses

    paths = plot_losses(cfg.paths.log_dir, cfg.paths.results_dir, cfg.language)
    log.info("Loss plots: %s", ", ".join(str(p) for p in paths))
    return paths


def resolve_checkpoint(cfg, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    for name in ("model_inference.pt", "best.pt", "last.pt"):
        if (ckpt_dir / name).exists():
            return ckpt_dir / name
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}; train first or pass --checkpoint.")


def load_test_sentences(cfg) -> list[dict]:
    path = Path(cfg.get("test_sentences", Path(cfg.project_root) / "data" / "test_sentences.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[cfg.language]


def cmd_synthesize(cfg, args=None):
    from .synthesize import Synthesizer
    from .utils.plots import plot_variance

    ckpt = resolve_checkpoint(cfg, getattr(args, "checkpoint", None) if args else None)
    synth = Synthesizer(ckpt, cfg.paths.get("vocoder_checkpoint"), cfg.train.get("device", "auto"))
    speaker = getattr(args, "speaker", None) if args else None
    controls = {
        "duration_control": float(getattr(args, "duration", 1.0) or 1.0) if args else 1.0,
        "pitch_control": float(getattr(args, "pitch", 1.0) or 1.0) if args else 1.0,
        "energy_control": float(getattr(args, "energy", 1.0) or 1.0) if args else 1.0,
    }
    results_dir = Path(cfg.paths.results_dir)
    text = getattr(args, "text", None) if args else None
    if text:
        out = ensure_dir(getattr(args, "out", None) or results_dir / "custom")
        res = synth.synthesize(text, speaker, **controls)
        name = getattr(args, "name", None) or "custom"
        res.save(out / f"{name}.wav", out / f"{name}.json")
        plot_variance({**res.to_dict(), "mel": res.mel}, name, out / "plots", _roman(res))
        log.info("Wrote %s (%.2fs audio)", out / f"{name}.wav", res.duration_sec)
        return out

    audio_dir = ensure_dir(results_dir / "test_generated_audios")
    plot_dir = ensure_dir(results_dir / "variance_adaptor_plots")
    summary = []
    for item in load_test_sentences(cfg):
        res = synth.synthesize(item["text"], speaker, **controls)
        res.save(audio_dir / f"{item['id']}.wav", audio_dir / f"{item['id']}.json")
        plot_variance({**res.to_dict(), "mel": res.mel}, item["id"], plot_dir, _roman(res))
        summary.append({"id": item["id"], "focus": item.get("focus", ""), **res.to_dict()})
        log.info("%s: %.2fs audio, %d tokens, %s", item["id"], res.duration_sec, len(res.phonemes), res.vocoder)
    (results_dir / "test_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return audio_dir


def _roman(res) -> list[str] | None:
    """Romanised word labels (phones joined) for scripts matplotlib cannot render."""
    if res.language != "telugu":
        return None
    return ["".join(res.phonemes[w["start"]:w["end"]]) for w in res.words]


def cmd_report(cfg, args=None):
    from .report.builder import build_report

    root = Path(cfg.project_root)
    configs = getattr(args, "configs", None) or [str(root / "English" / "config.yaml"), str(root / "Telugu" / "config.yaml")]
    return build_report([load_config(c) for c in configs], root / "Report")


COMMANDS = {
    "prepare": cmd_prepare, "preprocess": cmd_preprocess, "train": cmd_train,
    "plot": cmd_plot, "synthesize": cmd_synthesize, "report": cmd_report,
}


def cmd_pipeline(cfg, args):
    stages = args.stages.split(",") if args.stages else STAGES
    for stage in stages:
        stage = stage.strip()
        if stage not in COMMANDS:
            raise SystemExit(f"Unknown stage {stage!r}; choose from {STAGES}")
        log.info("========== %s | %s ==========", cfg.language, stage)
        COMMANDS[stage](cfg, args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fastspeech2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, help_text):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--config", required=True, help="Path to English/config.yaml or Telugu/config.yaml")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config value")
        return sp

    add("prepare", "Download HF dataset and select ~5h train / ~1h val")
    sp = add("preprocess", "Phonemise and extract mel/pitch/energy")
    sp.add_argument("--manifest", help="Custom manifest.jsonl (skips the prepare stage)")
    add("train", "Train the model")
    add("plot", "Plot training/validation losses")
    sp = add("synthesize", "Synthesise the test sentences (default) or --text")
    for flag in ("--checkpoint", "--text", "--speaker", "--out", "--name"):
        sp.add_argument(flag)
    for flag in ("--duration", "--pitch", "--energy"):
        sp.add_argument(flag, type=float, default=1.0)
    sp = add("report", "Build Report/Technical_Report.pdf from both languages' results")
    sp.add_argument("--configs", nargs="*")
    sp = add("pipeline", "Run several stages in order")
    sp.add_argument("--stages", default=",".join(STAGES))
    sp.add_argument("--manifest")
    sp.add_argument("--configs", nargs="*")
    for flag in ("--checkpoint", "--text", "--speaker", "--out", "--name"):
        sp.add_argument(flag)
    for flag in ("--duration", "--pitch", "--energy"):
        sp.add_argument(flag, type=float, default=1.0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = _cfg(args)
    if args.command == "pipeline":
        cmd_pipeline(cfg, args)
    else:
        COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
