# Deployment

The web studio (`app/`) is a FastAPI service with a static front-end. It loads
`English/checkpoints/model_inference.pt` and `Telugu/checkpoints/model_inference.pt` lazily and
runs comfortably on CPU (≈0.3–1.5 s per sentence).

## 1. Local

```bash
pip install -r requirements.txt
python scripts/download_assets.py          # vocoder + NLTK data
uvicorn app.server:app --port 7860         # http://localhost:7860 ; API docs at /docs
```

## 2. Docker / docker compose

```bash
docker compose up --build                  # mounts checkpoints, Results and Report from the host
# or
docker build -t fastspeech2-studio . && docker run -p 7860:7860 fastspeech2-studio
```

The image is CPU-only (~1.2 GB) and runs as a non-root user. If the checkpoints are not baked into the
image, set `FS2_ENGLISH_CHECKPOINT_URL` / `FS2_TELUGU_CHECKPOINT_URL`; they are downloaded at start-up.

## 3. Hugging Face Spaces (free, public URL)

1. Create a Space → SDK **Docker** → hardware *CPU basic*.
2. Push this repository to the Space (or connect the GitHub repo). Add to the top of `README.md` in the Space:

   ```yaml
   ---
   title: FastSpeech2 Studio
   emoji: 🗣️
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   ---
   ```
3. Upload the two `model_inference.pt` files (≈70 MB each) into `English/checkpoints/` and
   `Telugu/checkpoints/` of the Space, or host them in a model repo and set the `*_CHECKPOINT_URL`
   variables in the Space settings.

## 4. Render / Railway / Fly.io / any VM

Use the Dockerfile. The service listens on `$PORT` (default 7860) and exposes `/api/health` for health checks.

## Environment variables

| Variable | Purpose |
|---|---|
| `FS2_ENGLISH_CHECKPOINT`, `FS2_TELUGU_CHECKPOINT` | explicit local checkpoint paths |
| `FS2_ENGLISH_CHECKPOINT_URL`, `FS2_TELUGU_CHECKPOINT_URL` | download checkpoints at start-up |
| `FS2_VOCODER` | HiFi-GAN generator path (default `pretrained/hifigan_universal.pth.tar`) |
| `FS2_DEVICE` | `auto`, `cpu` or `cuda` |
| `FS2_PRELOAD=1` | load both models at start-up instead of on first request |
| `PORT` | listening port (Docker image) |

## API

```bash
curl -X POST localhost:7860/api/synthesize.wav -H 'Content-Type: application/json' \
     -d '{"language":"telugu","text":"తాడుపై అతని పట్టు చాలా బలంగా ఉంది.","duration_control":1.0}' -o out.wav
```

`POST /api/synthesize` returns JSON with base64 WAV, phonemes, per-token durations (frames and ms),
pitch (Hz), energy and a down-sampled mel spectrogram.
