"""End-to-end smoke test: synthetic data -> preprocess -> train -> synthesize -> plots."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMALL = [
    "train.total_steps=12", "train.log_every=4", "train.val_every=6", "train.save_every=6",
    "train.num_workers=0", "train.max_frames_per_batch=3000", "train.max_batch_size=8",
    "optimizer.warmup_steps=4", "model.aligner.bin_loss_start_step=4", "model.aligner.bin_loss_warmup_steps=4",
    "model.encoder.layers=1", "model.decoder.layers=1", "model.encoder.conv_filter_size=128",
    "model.decoder.conv_filter_size=128", "model.postnet.channels=64",
]


@pytest.mark.parametrize("lang,folder", [("english", "English"), ("telugu", "Telugu")])
def test_end_to_end(tmp_path, lang, folder):
    data = tmp_path / lang
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), "--language", lang,
                    "--out", str(data), "--n", "24"], check=True)
    sets = [f"paths.raw_dir={data}", f"paths.preprocessed_dir={data}/pre", f"paths.log_dir={data}/logs",
            f"paths.checkpoint_dir={data}/ckpt", f"paths.results_dir={data}/results",
            f"paths.vocoder_checkpoint={data}/no_vocoder.pt", *SMALL]
    args = [a for s in sets for a in ("--set", s)]
    run = lambda script, *extra: subprocess.run([sys.executable, str(ROOT / folder / script), *extra, *args],  # noqa: E731
                                                check=True, cwd=ROOT)
    run("preprocess.py", "--skip-download", "--workers", "1")
    run("train.py")
    assert (data / "ckpt" / "model_inference.pt").exists()
    assert (data / "results" / "training_loss.png").exists()
    assert (data / "results" / "validation_loss.png").exists()
    # vocoder path does not exist and download is attempted; Griffin-Lim fallback keeps the test offline-safe
    run("inference.py", "--set", "paths.vocoder_checkpoint=null")
    results = json.loads((data / "results" / "test_results.json").read_text(encoding="utf-8"))
    assert len(results) == 5
    for r in results:
        assert (data / "results" / "test_generated_audios" / f"{r['id']}.wav").exists()
        for kind in ("duration", "pitch", "energy"):
            assert (data / "results" / "variance_adaptor_plots" / kind / f"{r['id']}.png").exists()
        assert len(r["durations_frames"]) == len(r["phonemes"]) == len(r["pitch_hz"]) == len(r["energy"])
