# FastSpeech 2 — English & Telugu Text-to-Speech

Two independently trained [FastSpeech 2](https://arxiv.org/abs/2006.04558) models — **English** on
[Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (Indian-accented English) and **Telugu** on
[IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) — built on
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), with a fully automated pipeline, an
auto-generated technical report and a light-themed web studio for deployment.

| | |
|---|---|
| **Model** | FastSpeech 2 (4-layer FFT encoder, variance adaptor, 6-layer decoder, PostNet) + universal HiFi-GAN |
| **Alignment** | learned, MFA-free (RAD-TTS: CTC forward-sum + Monotonic Alignment Search) |
| **Text** | English: normaliser + g2p_en (POS-aware homographs) · Telugu: rule-based akshara G2P + Telugu number normaliser |
| **Data** | ~5 h train / ~1 h validation per language, deterministic, speaker-stratified, text-disjoint |
| **Automation** | `make all` → download → select → features → train → loss plots → test synthesis → variance plots → PDF report |
| **Deployment** | FastAPI + web UI, Docker, docker compose, Hugging Face Spaces |
| **Tests** | unit tests + end-to-end smoke test (synthetic data, CPU, ~2 min) in CI |

> **Status of trained artefacts.** The code, configs and automation are complete and tested end to end.
> The trained checkpoints, generated audio, loss/variance plots and the final PDF are *produced by
> running the pipeline* on a GPU (see [Quick start](#quick-start)); `Results/` and `*/checkpoints/` are
> empty until you do.

---

## Quick start

```bash
git clone <your-repo-url> fastspeech2 && cd fastspeech2
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                           # CUDA PyTorch: see pytorch.org first
python scripts/download_assets.py                         # HiFi-GAN vocoder + NLTK data

# Svarah is gated: accept its terms on Hugging Face, then
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

make all            # == bash scripts/run_all.sh  (both languages, end to end)
make serve          # web studio on http://localhost:7860
```

No local GPU? Open [`notebooks/Train_on_Colab.ipynb`](notebooks/Train_on_Colab.ipynb) — it keeps
everything on Google Drive and resumes automatically after a disconnect.

### Step by step (per language)

```bash
python Telugu/preprocess.py      # download, select 5 h/1 h, extract mel/pitch/energy
python Telugu/train.py           # train (resumes from checkpoints/last.pt) + training/validation loss plots
python Telugu/inference.py       # 5 test sentences -> audio + duration/pitch/energy plots
python Telugu/inference.py --text "మీకు స్వాగతం" --name welcome --duration 1.1 --pitch 1.05
# same for English/
python -m fastspeech2.cli report --config English/config.yaml   # Report/Technical_Report.pdf (+ .md)
```

Every config value can be overridden from the command line, e.g.
`python English/train.py --set train.total_steps=60000 --set train.max_frames_per_batch=8000`.

### Expected compute

| Stage | Hardware | Time (approx.) |
|---|---|---|
| Download + selection + features | CPU (4 cores) | 15–30 min per language |
| Training, 40 k steps | T4 / L4 16 GB | ~3–5 h per language (check `sec_per_step` in `logs/<lang>/train_log.csv`) |
| Inference | CPU | ~0.3–1.5 s per sentence |

Out of memory? Lower `train.max_frames_per_batch` (e.g. 8000). Short on time? `STEPS=20000 make all`;
FastSpeech 2 is usually intelligible after ~20 k steps on 5 h of data.

---

## Project structure

```
project/
├── English/                 train.py · preprocess.py · inference.py · config.yaml · checkpoints/
├── Telugu/                  train.py · preprocess.py · inference.py · config.yaml · checkpoints/
├── Results/<Language>/      training_loss.png · validation_loss.png · test_generated_audios/
│                            variance_adaptor_plots/{duration,pitch,energy,overview}/
├── Report/                  Technical_Report.pdf (generated) · observations.yaml (your listening notes)
├── fastspeech2/             the library — same layout/file names as ming024/FastSpeech2
│   ├── transformer/         Constants.py · SubLayers.py · Layers.py · Models.py
│   ├── model/               fastspeech2.py · modules.py · loss.py · optimizer.py · aligner.py (new)
│   ├── text/                symbols.py · cleaners.py · numbers.py · english.py · telugu.py (new) · numbers_te.py (new)
│   ├── audio/               audio_processing.py (mel / WORLD pitch / energy, HiFi-GAN compatible)
│   ├── preprocessor/        preprocessor.py · hf_subset.py (HF download + 5 h / 1 h selection)
│   ├── hifigan/             models.py · config.json
│   ├── utils/               model.py · tools.py · plots.py · config.py · misc.py
│   ├── dataset.py · train.py · evaluate.py · synthesize.py
│   ├── report/              analysis.py · builder.py (new)
│   └── cli.py               one entry point for every stage
├── app/                     FastAPI server + static web UI (light theme)
├── data/test_sentences.json the assignment's E1–E4 / T1–T4 sentences
├── scripts/                 run_all.sh · download_assets.py · make_synthetic_data.py
├── tests/                   unit tests + end-to-end smoke test
├── notebooks/               Train_on_Colab.ipynb
├── docs/                    MODIFICATIONS.md · DEPLOYMENT.md
├── Dockerfile · docker-compose.yml · Makefile · requirements.txt · pyproject.toml
└── README.md
```

## How the assignment requirements map to the code

| Requirement | Where |
|---|---|
| Separate English & Telugu models | `English/config.yaml`, `Telugu/config.yaml`; separate checkpoints/logs/results |
| ~5 h train / ~1 h val, disjoint, documented strategy | `fastspeech2/preprocessor/hf_subset.py` → `data/<lang>/raw/selection_report.json`; asserted + unit-tested |
| Modified files from the reference implementation | `fastspeech2/` keeps ming024's file names; each file has a `Modified from …` header; full table in [`docs/MODIFICATIONS.md`](docs/MODIFICATIONS.md) |
| Telugu support | `fastspeech2/text/telugu.py`, `numbers_te.py`, learned aligner |
| Training / validation loss plots | `Results/<Lang>/training_loss.png`, `validation_loss.png` |
| Audio for all test sentences | `Results/<Lang>/test_generated_audios/<ID>.wav` (+ `.json` with all predictions) |
| Duration / pitch / energy plots per sentence | `Results/<Lang>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png` |
| Technical report | `Report/Technical_Report.pdf`, generated from the actual logs/results; listening notes from `Report/observations.yaml` |
| Reproducibility | fixed seeds, pinned requirements, configs, `make` targets, CI |

### Data selection (both languages)

1. Metadata pass over the whole split (text, gender/speaker, duration).
2. Keep 1–15 s clips whose text passes a script check; relax the 15 s limit only if needed.
3. De-duplicate on normalised text → no prompt appears in both splits.
4. Speakers: Telugu keeps one speaker (female preferred) when it alone covers 6 h, else both;
   English keeps all Svarah speakers (~5 min each) and learns a speaker embedding.
5. Validation (1 h) is drawn first, round-robin across speakers; training (5 h) from the remainder.
6. Decode → 22.05 kHz → trim silence → peak-normalise → 16-bit WAV + `manifest.jsonl`.

### Test sentences

`data/test_sentences.json` holds E1–E4 and T1–T4 exactly as given. The Telugu lines were re-typed
because PDF text extraction breaks Telugu conjuncts; the intentional split *అడ విలో* (T3, mirroring
E3 *rhyth m*) is preserved. **Check them against the assignment PDF before submitting.**

### Report workflow

The report is regenerated from real artefacts each time you run `make report`: dataset statistics,
configurations, loss-curve diagnostics (convergence step, spikes, over-fitting check, generalisation
gap), and per-sentence variance analysis (per-word durations, F0, energy, declination, pause lengths,
homograph stress for E2). Only *you* can listen to the audio, so write what you hear into
`Report/observations.yaml` and rebuild — sections still marked TODO are shown as "pending".

## Web studio & API

`make serve` (or `docker compose up`) → http://localhost:7860

* **Studio** — type or pick a test sentence, choose speaker, adjust speed / pitch / energy, listen and
  download; view the mel spectrogram, interactive duration/pitch/energy charts and per-word phonemes.
* **Assignment results** — loss curves, test audios and variance plots straight from `Results/`.
* **API** — `POST /api/synthesize`, `POST /api/synthesize.wav`, `GET /api/models`, OpenAPI docs at `/docs`.

Deployment to Hugging Face Spaces, Render or any VM: see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Checkpoints

Training writes `step_*.pt`, `last.pt` and `best.pt` (weights + optimiser, ~400 MB, git-ignored) and
exports **`model_inference.pt`** (fp16 weights + vocabulary + speakers + statistics + config, ~70 MB)
which is under GitHub's 100 MB limit and is all the web app needs. To also share training checkpoints,
use Git LFS (`git lfs track "*.pt"`) or upload them to a Hugging Face model repo and link them here:

| Model | Download |
|---|---|
| English | `TODO: link` |
| Telugu | `TODO: link` |

## Testing

```bash
pytest -q          # text frontends, data selection, end-to-end pipeline on synthetic data (CPU)
```

## Troubleshooting

* **`401`/`GatedRepoError` for Svarah** — accept the dataset terms on Hugging Face and `export HF_TOKEN=...`.
* **Telugu shows boxes in plots** — plots use romanised phone labels on purpose; Telugu script is shown in the web UI and Markdown report.
* **`pyworld` fails to build** — `pip install numpy cython` first, or use Python 3.10/3.11 where wheels exist.
* **Vocoder download blocked** — the downloader falls back to a sparse `git clone` of the reference repo; if both fail, Griffin-Lim is used.

## References

* Y. Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*, ICLR 2021.
* ming024/FastSpeech2 — reference PyTorch implementation (MIT).
* R. Badlani et al., *One TTS Alignment To Rule Them All*, ICASSP 2022.
* J. Kong et al., *HiFi-GAN*, NeurIPS 2020.
* T. Javed et al., *Svarah: Evaluating English ASR Systems on Indian Accents*, Interspeech 2023.
