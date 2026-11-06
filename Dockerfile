# FastSpeech 2 Studio — CPU inference image (works on Hugging Face Spaces, Render, Railway, Fly, any VM).
#   docker build -t fastspeech2-studio .
#   docker run -p 7860:7860 fastspeech2-studio
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl \
    NLTK_DATA=/usr/share/nltk_data \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-deploy.txt \
    && python -c "import nltk; [nltk.download(p, download_dir='/usr/share/nltk_data', quiet=True) for p in ('averaged_perceptron_tagger_eng','averaged_perceptron_tagger','cmudict','punkt_tab')]"

COPY . .
# Fetch the universal HiFi-GAN at build time (falls back to Griffin-Lim if this fails)
RUN python scripts/download_assets.py --vocoder || true

# Non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fs http://localhost:${PORT}/api/health || exit 1
CMD ["sh", "-c", "python scripts/download_assets.py --checkpoints || true; exec uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1"]
