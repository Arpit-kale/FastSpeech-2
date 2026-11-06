"""Modified from ming024/FastSpeech2 ``audio/stft.py`` + ``audio/tools.py`` + ``audio/audio_processing.py``.

Audio I/O and acoustic feature extraction.

The mel spectrogram matches the HiFi-GAN (jik876) recipe used to train the
universal vocoder shipped with ming024/FastSpeech2: 22.05 kHz, n_fft 1024,
hop 256, 80 mels, fmax 8 kHz, magnitude STFT with (n_fft-hop)/2 reflect
padding, natural-log compression clamped at 1e-5. Pitch comes from WORLD
(DIO + StoneMask) on the same hop; energy is the L2 norm of each STFT frame.
"""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


class AudioProcessor:
    def __init__(self, cfg: dict):
        self.sr = int(cfg["sampling_rate"])
        self.n_fft = int(cfg["n_fft"])
        self.hop = int(cfg["hop_length"])
        self.win = int(cfg["win_length"])
        self.n_mels = int(cfg["n_mels"])
        self.fmin = float(cfg["mel_fmin"])
        self.fmax = float(cfg["mel_fmax"])
        self.trim_top_db = float(cfg.get("trim_top_db", 40))
        self.trim_margin = float(cfg.get("trim_margin_sec", 0.05))
        self.f0_floor = float(cfg.get("f0_min", 60.0))
        self.f0_ceil = float(cfg.get("f0_max", 600.0))
        self._mel_basis = torch.from_numpy(
            librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        ).float()
        self._window = torch.hann_window(self.win)

    # ------------------------------------------------------------------ I/O
    def load(self, source: str | Path | bytes, trim: bool = True) -> np.ndarray:
        """Load mono float32 audio at ``self.sr`` from a path or raw bytes."""
        if isinstance(source, (bytes, bytearray)):
            try:
                wav, sr = sf.read(io.BytesIO(source), dtype="float32", always_2d=True)
                wav = wav.mean(axis=1)
            except Exception:  # e.g. mp3 with an old libsndfile
                wav, sr = librosa.load(io.BytesIO(source), sr=None, mono=True)
        else:
            wav, sr = librosa.load(str(source), sr=None, mono=True)
        if sr != self.sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr, res_type="soxr_hq")
        if trim:
            wav = self.trim_silence(wav)
        peak = np.abs(wav).max() if wav.size else 0.0
        if peak > 0:
            wav = wav / peak * 0.95
        return wav.astype(np.float32)

    def trim_silence(self, wav: np.ndarray) -> np.ndarray:
        if wav.size == 0:
            return wav
        _, (start, end) = librosa.effects.trim(wav, top_db=self.trim_top_db, frame_length=1024, hop_length=256)
        margin = int(self.trim_margin * self.sr)
        return wav[max(0, start - margin): min(len(wav), end + margin)]

    def save(self, wav: np.ndarray, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.clip(wav, -1.0, 1.0), self.sr, subtype="PCM_16")

    @staticmethod
    def duration_of_bytes(data: bytes) -> float | None:
        try:
            info = sf.info(io.BytesIO(data))
            return info.frames / float(info.samplerate)
        except Exception:
            return None

    # ------------------------------------------------------------- features
    def mel_and_energy(self, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (log-mel [n_mels, T], energy [T]); T == len(wav) // hop."""
        y = torch.from_numpy(wav).float().unsqueeze(0)
        pad = (self.n_fft - self.hop) // 2
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(
            y, self.n_fft, hop_length=self.hop, win_length=self.win, window=self._window,
            center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True,
        )
        mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)[0]  # [F, T]
        mel = torch.log(torch.clamp(self._mel_basis @ mag, min=1e-5))
        energy = torch.linalg.vector_norm(mag, dim=0)
        return mel.numpy().astype(np.float32), energy.numpy().astype(np.float32)

    def pitch(self, wav: np.ndarray, n_frames: int) -> np.ndarray:
        """Raw F0 in Hz (0 where unvoiced), length ``n_frames``."""
        import pyworld as pw

        x = wav.astype(np.float64)
        frame_period = self.hop / self.sr * 1000.0
        f0, t = pw.dio(x, self.sr, f0_floor=self.f0_floor, f0_ceil=self.f0_ceil, frame_period=frame_period)
        f0 = pw.stonemask(x, f0, t, self.sr)
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)))
        return f0[:n_frames].astype(np.float32)

    @staticmethod
    def interpolate_f0(f0: np.ndarray) -> np.ndarray:
        """Linearly interpolate across unvoiced regions (edges held constant)."""
        voiced = np.nonzero(f0 > 0)[0]
        if len(voiced) == 0:
            return f0.copy()
        return np.interp(np.arange(len(f0)), voiced, f0[voiced]).astype(np.float32)
