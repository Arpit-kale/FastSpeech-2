"""FastAPI service + web UI for the English and Telugu FastSpeech 2 models.

    uvicorn app.server:app --host 0.0.0.0 --port 7860

Environment variables (all optional):
    FS2_ENGLISH_CHECKPOINT / FS2_TELUGU_CHECKPOINT   explicit checkpoint paths
    FS2_VOCODER                                      HiFi-GAN generator path
    FS2_DEVICE                                       auto | cpu | cuda
    FS2_PRELOAD=1                                    load models at start-up
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
LANGS = {"english": "English", "telugu": "Telugu"}
MAX_CHARS = 600


def _checkpoint_for(lang: str) -> Path | None:
    env = os.environ.get(f"FS2_{lang.upper()}_CHECKPOINT")
    if env:
        return Path(env) if Path(env).exists() else None
    ckpt_dir = ROOT / LANGS[lang] / "checkpoints"
    for name in ("model_inference.pt", "best.pt", "last.pt"):
        if (ckpt_dir / name).exists():
            return ckpt_dir / name
    return None


class ModelRegistry:
    """Lazily loads one Synthesizer per language (thread-safe)."""

    def __init__(self):
        self._models: dict = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self.vocoder = os.environ.get("FS2_VOCODER", str(ROOT / "pretrained" / "hifigan_universal.pth.tar"))
        self.device = os.environ.get("FS2_DEVICE", "auto")

    def get(self, lang: str):
        if lang not in LANGS:
            raise HTTPException(404, f"Unknown language '{lang}'")
        with self._lock:
            if lang in self._models:
                return self._models[lang]
            ckpt = _checkpoint_for(lang)
            if ckpt is None:
                raise HTTPException(503, f"No trained {LANGS[lang]} checkpoint found. Train the model "
                                         f"({LANGS[lang]}/train.py) or set FS2_{lang.upper()}_CHECKPOINT.")
            try:
                from fastspeech2.synthesize import Synthesizer

                self._models[lang] = Synthesizer(ckpt, self.vocoder, self.device)
                self._errors.pop(lang, None)
            except Exception as exc:  # surface loading problems to the UI
                self._errors[lang] = str(exc)
                raise HTTPException(500, f"Failed to load {lang} model: {exc}") from exc
            return self._models[lang]

    def status(self) -> list[dict]:
        out = []
        for lang, folder in LANGS.items():
            ckpt = _checkpoint_for(lang)
            m = self._models.get(lang)
            out.append({
                "language": lang, "label": folder, "available": ckpt is not None, "loaded": m is not None,
                "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt and ckpt.is_relative_to(ROOT) else (str(ckpt) if ckpt else None),
                "speakers": m.speakers if m else [], "default_speaker": m.default_speaker if m else None,
                "step": m.meta.get("step") if m else None, "vocoder": m.vocoder.kind if m else None,
                "error": self._errors.get(lang),
            })
        return out


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("FS2_PRELOAD") == "1":
        for lang in LANGS:
            try:
                registry.get(lang)
            except HTTPException:
                pass
    yield


app = FastAPI(title="FastSpeech 2 — English & Telugu", version="1.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SynthesisRequest(BaseModel):
    language: str = Field("english", pattern="^(english|telugu)$")
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)
    speaker: str | None = None
    duration_control: float = Field(1.0, ge=0.5, le=2.0)
    pitch_control: float = Field(1.0, ge=0.5, le=2.0)
    energy_control: float = Field(1.0, ge=0.5, le=2.0)


def _wav_b64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mel_preview(mel: np.ndarray, max_frames: int = 400) -> list[list[float]]:
    """Down-sampled mel (frames x 40 bins) for a light-weight spectrogram view."""
    step = max(1, int(np.ceil(len(mel) / max_frames)))
    m = mel[::step]
    m = m.reshape(len(m), 40, -1).mean(-1) if m.shape[1] % 40 == 0 else m
    lo, hi = float(m.min()), float(m.max())
    return np.round((m - lo) / max(1e-6, hi - lo), 3).tolist()


@app.get("/api/health")
def health():
    return {"status": "ok", "models": registry.status()}


@app.get("/api/models")
def models():
    return registry.status()


@app.get("/api/test-sentences")
def test_sentences():
    data = json.loads((ROOT / "data" / "test_sentences.json").read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


@app.post("/api/load/{language}")
def load(language: str):
    registry.get(language)
    return next(s for s in registry.status() if s["language"] == language)


@app.post("/api/synthesize")
def synthesize(req: SynthesisRequest):
    synth = registry.get(req.language)
    try:
        res = synth.synthesize(req.text, req.speaker, req.duration_control, req.pitch_control, req.energy_control)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    payload = res.to_dict()
    payload["audio_wav_base64"] = _wav_b64(res.wav, res.sampling_rate)
    payload["mel_preview"] = _mel_preview(res.mel)
    return JSONResponse(payload)


@app.post("/api/synthesize.wav")
def synthesize_wav(req: SynthesisRequest):
    synth = registry.get(req.language)
    res = synth.synthesize(req.text, req.speaker, req.duration_control, req.pitch_control, req.energy_control)
    buf = io.BytesIO()
    sf.write(buf, res.wav, res.sampling_rate, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"Content-Disposition": 'attachment; filename="fastspeech2.wav"'})


@app.get("/api/results/{language}")
def results(language: str):
    """Pre-computed assignment artefacts (loss plots, test audios, variance plots)."""
    if language not in LANGS:
        raise HTTPException(404)
    base = ROOT / "Results" / LANGS[language]
    url = lambda p: f"/results/{p.relative_to(ROOT / 'Results').as_posix()}"  # noqa: E731
    items = []
    summary = base / "test_results.json"
    meta = {r["id"]: r for r in json.loads(summary.read_text(encoding="utf-8"))} if summary.exists() else {}
    for wav in sorted((base / "test_generated_audios").glob("*.wav")):
        sid = wav.stem
        plots = {k: url(p) for k in ("duration", "pitch", "energy", "overview")
                 if (p := base / "variance_adaptor_plots" / k / f"{sid}.png").exists()}
        items.append({"id": sid, "audio": url(wav), "plots": plots, "text": meta.get(sid, {}).get("text"),
                      "focus": meta.get(sid, {}).get("focus")})
    losses = {n: url(base / f"{n}.png") for n in ("training_loss", "validation_loss") if (base / f"{n}.png").exists()}
    return {"language": language, "losses": losses, "tests": items}


(ROOT / "Results").mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=ROOT / "Results"), name="results")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/report")
def report():
    pdf = ROOT / "Report" / "Technical_Report.pdf"
    if not pdf.exists():
        raise HTTPException(404, "Report not built yet (make report)")
    return FileResponse(pdf, media_type="application/pdf")
