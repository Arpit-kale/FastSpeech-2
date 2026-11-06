"""Modified from ming024/FastSpeech2 ``utils/model.py``.

``get_model`` / ``get_vocoder`` / ``vocoder_infer`` of the reference become:

* ``build_model`` / ``get_model`` - construct FastSpeech 2 from config + data sizes;
* ``save_checkpoint`` / ``export_for_inference`` / ``load_inference_checkpoint`` -
  checkpoints now embed config, vocabulary, speaker map and pitch/energy
  statistics, and the exported fp16 file (~70 MB, aligner stripped) is
  self-contained and under GitHub's 100 MB limit;
* ``Vocoder`` (= get_vocoder + vocoder_infer) - HiFi-GAN with automatic download
  of the universal generator and a Griffin-Lim fallback.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

from ..hifigan import V1_CONFIG, Generator
from ..model import FastSpeech2
from .config import to_plain
from .misc import get_logger

log = get_logger("model")
REFERENCE_REPO = "https://github.com/ming024/FastSpeech2.git"
UNIVERSAL_URL = "https://github.com/ming024/FastSpeech2/raw/master/hifigan/generator_universal.pth.tar.zip"


def build_model(model_cfg: dict, n_symbols: int, n_speakers: int, stats: dict, n_mels: int) -> FastSpeech2:
    return FastSpeech2(to_plain(model_cfg), n_symbols, n_speakers, stats, n_mels)


def save_checkpoint(path: str | Path, model, optimizer, scheduler, scaler, step: int, meta: dict, best_val: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best_val": best_val,
        "meta": meta,
    }, tmp)
    tmp.replace(path)


def export_for_inference(train_ckpt: str | Path, out_path: str | Path, half: bool = True) -> Path:
    """Strip optimizer state and the aligner (training-only), store weights in fp16.

    The result (~70 MB for the paper-size model) stays under GitHub's 100 MB
    file limit and contains everything needed to synthesise: weights, config,
    vocabulary, speaker map and pitch/energy statistics.
    """
    ckpt = torch.load(train_ckpt, map_location="cpu", weights_only=False)
    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("aligner.")}
    if half:
        state = {k: (v.half() if v.is_floating_point() and "bins" not in k else v) for k, v in state.items()}
    meta = dict(ckpt["meta"])
    meta["step"] = ckpt["step"]
    meta["best_val"] = ckpt.get("best_val")
    out_path = Path(out_path)
    torch.save({"model": state, "meta": meta, "format": "fastspeech2-inference-v1"}, out_path)
    log.info("Exported inference checkpoint -> %s (%.1f MB)", out_path, out_path.stat().st_size / 2**20)
    return out_path


def load_inference_checkpoint(path: str | Path, device: torch.device):
    """Load either a training or an exported checkpoint; returns (model, meta)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["model_cfg"], len(meta["vocab"]), max(1, len(meta["speaker_map"])),
                        meta["stats"], int(meta["audio_cfg"]["n_mels"]))
    state = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("aligner.")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    return model, meta


get_model = build_model


def _extract_zip(payload: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".pth.tar"))
        dest.write_bytes(zf.read(name))


def _download_via_git(dest: Path) -> None:
    """Sparse-clone only the ``hifigan/`` folder of the reference repo."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "fs2"
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REFERENCE_REPO, str(repo)],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "hifigan"], check=True, capture_output=True)
        _extract_zip((repo / "hifigan" / "generator_universal.pth.tar.zip").read_bytes(), dest)


def download_universal_hifigan(dest: str | Path) -> Path:
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading universal HiFi-GAN from %s", UNIVERSAL_URL)
    try:
        with urllib.request.urlopen(UNIVERSAL_URL, timeout=120) as resp:
            _extract_zip(resp.read(), dest)
    except Exception as exc:
        log.warning("Direct download failed (%s); trying a sparse git clone.", exc)
        _download_via_git(dest)
    return dest


class Vocoder:
    """Mel -> waveform. Uses HiFi-GAN when a checkpoint is available, else Griffin-Lim."""

    def __init__(self, checkpoint: str | Path | None, audio_cfg: dict, device: torch.device, auto_download: bool = True):
        self.device = device
        self.audio_cfg = audio_cfg
        self.model = None
        self.kind = "griffin-lim"
        path = Path(checkpoint) if checkpoint else None
        if path is not None and not path.exists() and auto_download:
            try:
                download_universal_hifigan(path)
            except Exception as exc:  # network not available etc.
                log.warning("HiFi-GAN download failed (%s); falling back to Griffin-Lim.", exc)
        if path is not None and path.exists():
            gen = Generator(V1_CONFIG, int(audio_cfg["n_mels"]))
            state = torch.load(path, map_location="cpu", weights_only=False)
            gen.load_state_dict(state["generator"] if "generator" in state else state)
            gen.eval()
            gen.remove_weight_norm()
            self.model = gen.to(device)
            self.kind = "hifigan"
        log.info("Vocoder: %s", self.kind)

    @torch.no_grad()
    def __call__(self, mel: torch.Tensor) -> np.ndarray:
        """mel: [T, n_mels] or [1, T, n_mels] natural-log magnitude mel."""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if self.model is not None:
            wav = self.model(mel.transpose(1, 2).to(self.device).float()).squeeze().cpu().numpy()
        else:
            wav = self._griffin_lim(mel[0].cpu().numpy())
        peak = np.abs(wav).max() if wav.size else 0.0
        if peak > 0.99:
            wav = wav / peak * 0.99
        return wav.astype(np.float32)

    def _griffin_lim(self, mel: np.ndarray) -> np.ndarray:
        import librosa

        a = self.audio_cfg
        mag_mel = np.exp(mel.T)
        return librosa.feature.inverse.mel_to_audio(
            mag_mel, sr=a["sampling_rate"], n_fft=a["n_fft"], hop_length=a["hop_length"],
            win_length=a["win_length"], fmin=a["mel_fmin"], fmax=a["mel_fmax"], power=1.0, n_iter=48,
        )
