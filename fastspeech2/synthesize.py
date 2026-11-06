"""Modified from ming024/FastSpeech2 ``synthesize.py``.

Text -> waveform. Changes: language-aware frontend (English/Telugu) instead of
``preprocess_english``/``preprocess_mandarin``; pitch control in Hz per speaker;
returns every variance-adaptor prediction (durations, pitch, energy, word
spans) for plotting, the report and the web API; vocoder fallback.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .text import Vocabulary, get_frontend
from .text.base import TextAnalysis
from .utils.model import Vocoder, load_inference_checkpoint
from .utils import get_device, get_logger

log = get_logger("synth")


@dataclass
class SynthesisResult:
    text: str
    normalized: str
    language: str
    speaker: str
    phonemes: list[str]
    words: list[dict]
    durations: list[int]  # frames per token
    pitch_hz: list[float]  # token-level predicted F0
    pitch_norm: list[float]
    energy: list[float]  # token-level predicted energy (denormalised L2 norm)
    energy_norm: list[float]
    sampling_rate: int
    hop_length: int
    wav: np.ndarray = field(repr=False)
    mel: np.ndarray = field(repr=False)
    vocoder: str = "hifigan"
    inference_ms: float = 0.0

    @property
    def frame_sec(self) -> float:
        return self.hop_length / self.sampling_rate

    @property
    def duration_sec(self) -> float:
        return len(self.wav) / self.sampling_rate

    def to_dict(self) -> dict:
        return {
            "text": self.text, "normalized": self.normalized, "language": self.language,
            "speaker": self.speaker, "phonemes": self.phonemes, "words": self.words,
            "durations_frames": self.durations,
            "durations_ms": [round(d * self.frame_sec * 1000, 1) for d in self.durations],
            "pitch_hz": [round(p, 2) for p in self.pitch_hz], "pitch_norm": [round(p, 4) for p in self.pitch_norm],
            "energy": [round(e, 4) for e in self.energy], "energy_norm": [round(e, 4) for e in self.energy_norm],
            "sampling_rate": self.sampling_rate, "hop_length": self.hop_length,
            "audio_seconds": round(self.duration_sec, 3), "vocoder": self.vocoder,
            "inference_ms": round(self.inference_ms, 1),
        }

    def save(self, wav_path: str | Path, json_path: str | Path | None = None) -> None:
        import soundfile as sf

        wav_path = Path(wav_path)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav_path), self.wav, self.sampling_rate, subtype="PCM_16")
        if json_path:
            Path(json_path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")


class Synthesizer:
    def __init__(self, checkpoint: str | Path, vocoder_checkpoint: str | Path | None = None,
                 device: str = "auto", auto_download_vocoder: bool = True):
        self.device = get_device(device)
        self.model, self.meta = load_inference_checkpoint(checkpoint, self.device)
        self.language = self.meta["language"]
        self.frontend = get_frontend(self.language)
        self.vocab = Vocabulary(self.meta["vocab"])
        self.speaker_map: dict[str, int] = self.meta["speaker_map"]
        self.stats = self.meta["stats"]
        self.audio_cfg = self.meta["audio_cfg"]
        self.vocoder = Vocoder(vocoder_checkpoint, self.audio_cfg, self.device, auto_download=auto_download_vocoder)
        self.default_speaker = self.stats.get("default_speaker") or next(iter(self.speaker_map), "default")
        log.info("Loaded %s model (step %s) with %d speaker(s)", self.language, self.meta.get("step"), len(self.speaker_map))

    @property
    def speakers(self) -> list[str]:
        return list(self.speaker_map)

    def _pitch_stats(self, speaker: str) -> tuple[float, float]:
        per = self.stats["pitch"]["per_speaker_hz"]
        return tuple(per.get(speaker) or next(iter(per.values())))

    def synthesize(self, text: str, speaker: str | None = None, duration_control: float = 1.0,
                   pitch_control: float = 1.0, energy_control: float = 1.0) -> SynthesisResult:
        """``duration_control`` > 1 slows speech; ``pitch_control`` scales F0 in Hz
        (e.g. 1.2 = 20 % higher); ``energy_control`` scales normalised energy."""
        t0 = time.perf_counter()
        analysis: TextAnalysis = self.frontend.analyze(text)
        phonemes = [p for p in analysis.phonemes if p in self.vocab]
        if len(phonemes) < 2:
            raise ValueError("Text produced no pronounceable tokens for this language.")
        speaker = speaker if speaker in self.speaker_map else self.default_speaker
        spk_id = self.speaker_map.get(speaker, 0)
        mean, std = self._pitch_stats(speaker)

        def p_transform(p_norm: torch.Tensor) -> torch.Tensor:
            if pitch_control == 1.0:
                return p_norm
            return ((p_norm * std + mean) * pitch_control - mean) / std

        tokens = torch.tensor([self.vocab.encode(phonemes)], device=self.device)
        speakers = torch.tensor([spk_id], device=self.device)
        out = self.model.infer(tokens, speakers, d_control=duration_control, p_transform=p_transform,
                               e_control=energy_control)
        mel = out["mel_post"][0, : int(out["mel_lens"][0])]
        wav = self.vocoder(mel)

        durations = out["durations"][0].long().cpu().tolist()
        p_norm = out["pitch_pred"][0].float().cpu().numpy()
        e_norm = out["energy_pred"][0].float().cpu().numpy()
        e_stats = self.stats["energy"]
        words, idx = [], 0
        for w in analysis.words:
            kept = [p for p in w.phones if p in self.vocab]
            words.append({"word": w.word, "start": idx, "end": idx + len(kept)})
            idx += len(kept)
        return SynthesisResult(
            text=text, normalized=analysis.normalized, language=self.language, speaker=speaker,
            phonemes=phonemes, words=words, durations=durations,
            pitch_hz=(p_norm * std + mean).tolist(), pitch_norm=p_norm.tolist(),
            energy=(e_norm * e_stats["std"] + e_stats["mean"]).tolist(), energy_norm=e_norm.tolist(),
            sampling_rate=int(self.audio_cfg["sampling_rate"]), hop_length=int(self.audio_cfg["hop_length"]),
            wav=wav, mel=mel.float().cpu().numpy(), vocoder=self.vocoder.kind,
            inference_ms=(time.perf_counter() - t0) * 1000,
        )
