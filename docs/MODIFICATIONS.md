# Changes relative to ming024/FastSpeech2

The reference repository (https://github.com/ming024/FastSpeech2) was used as the starting point. Its
model definition, variance adaptor design, loss functions and the HiFi-GAN vocoder interface are kept;
the code was reorganised into the `fastspeech2/` package and changed as below.

## Mapping of files

The package `fastspeech2/` mirrors the reference repository's layout, so every reference file that was
modified exists under the **same path and name**. Each file starts with a header stating
`Modified from ming024/FastSpeech2 <file>` (and what changed) or `NEW FILE`.

| ming024/FastSpeech2 | This project (`fastspeech2/…`) | Change |
|---|---|---|
| `transformer/Constants.py` | `transformer/Constants.py` | trimmed to PAD |
| `transformer/SubLayers.py` | `transformer/SubLayers.py` | custom MHA → `nn.MultiheadAttention`; `PositionwiseFeedForward` kept |
| `transformer/Layers.py` | `transformer/Layers.py` | `FFTBlock`, `PostNet` (sizes from config) |
| `transformer/Models.py` | `transformer/Models.py` | Encoder/Decoder share `FFTStack`; on-the-fly positions (no max length); embedding moved to model |
| `transformer/Modules.py` | — | removed (scaled dot-product attention now inside PyTorch) |
| `model/fastspeech2.py` | `model/fastspeech2.py` | integrated learned aligner; durations/pitch/energy targets from MAS; `forward` + `infer` |
| `model/modules.py` | `model/modules.py` | variance adaptor: phoneme-level pitch/energy, bins as buffers, F0 control in Hz, ≥1 frame per token |
| `model/loss.py` | `model/loss.py` | + CTC forward-sum and binarisation alignment losses |
| `model/optimizer.py` | `model/optimizer.py` | AdamW + Noam `LambdaLR` (resumable) |
| — | `model/aligner.py` | **new**: RAD-TTS aligner, beta-binomial prior, numba MAS |
| `text/symbols.py`, `text/__init__.py` | `text/symbols.py` | per-language vocabulary stored in checkpoint |
| `text/cleaners.py` | `text/cleaners.py` | `english_cleaners` + new `telugu_cleaners` |
| `text/numbers.py` | `text/numbers.py` | + times, dates, years, currency, ordinals |
| `text/cmudict.py`, `synthesize.py::preprocess_english` | `text/english.py` | g2p_en on whole sentences (POS homographs), per-word phones |
| — | `text/telugu.py`, `text/numbers_te.py`, `text/base.py` | **new**: Telugu G2P and normaliser, frontend interface |
| `audio/stft.py`, `audio/tools.py`, `audio/audio_processing.py` | `audio/audio_processing.py` | same mel recipe (HiFi-GAN compatible); WORLD F0; resampling/trim |
| `prepare_align.py`, `preprocessor/ljspeech.py` | `preprocessor/hf_subset.py` | Hugging Face download + deterministic 5 h / 1 h selection |
| `preprocessor/preprocessor.py` | `preprocessor/preprocessor.py` | no TextGrids; frame-level features; per-speaker pitch stats |
| `dataset.py` | `dataset.py` | frame-budget batch sampler; alignment prior in `collate` |
| `train.py` | `train.py` | AMP, resume, best-by-val, CSV/TensorBoard, export |
| `evaluate.py` | `evaluate.py` | full-set validation + alignment figure |
| `synthesize.py` | `synthesize.py` | language-aware; returns all variance predictions |
| `utils/model.py` | `utils/model.py` | self-contained checkpoints; vocoder auto-download + Griffin-Lim fallback |
| `utils/tools.py` | `utils/tools.py`, `utils/plots.py` | masks/padding kept; plots extended |
| `hifigan/models.py`, `hifigan/config.json` | `hifigan/models.py`, `hifigan/config.json` | Generator only (same parameter names) |
| `config/LJSpeech/{preprocess,model,train}.yaml` | `English/config.yaml`, `Telugu/config.yaml` | one file per language |
| `preprocess.py`, `train.py`, `synthesize.py` (root) | `English/*.py`, `Telugu/*.py` | thin per-language entry points |

## Why a learned aligner instead of MFA

* MFA has no pretrained Telugu acoustic model or pronunciation dictionary.
* Svarah is Indian-accented, partly spontaneous speech; the US-English MFA model aligns it poorly.
* Learning the alignment jointly removes an external, hard-to-install dependency and makes the whole
  pipeline a single `python preprocess.py && python train.py`.

The aligner follows *One TTS Alignment To Rule Them All* (Badlani et al., 2022): soft alignment from
L2 distances between conv-encoded text embeddings and mel frames, multiplied by a beta-binomial prior;
the CTC forward-sum loss trains it, MAS converts it to hard durations (duration targets + length
regulation through the hard alignment matrix), and a binarisation loss is ramped in from step 3 000.
Phoneme-level pitch/energy targets are the means of the frame values over each phoneme's aligned frames.

## English-specific

* Text normalisation: `10:45 AM` → *ten forty five ay em*, `12 November 2026` → *the twelfth of November
  twenty twenty six*, ordinals, decimals, currency (₹, $), percentages, common abbreviations.
* g2p_en is run on the full normalised sentence so its POS tagger can choose *CON-duct* (noun) vs
  *con-DUCT* (verb) for E2; per-word phones are recovered for analysis and plotting.
* Multi-speaker (Svarah has ~5 minutes per speaker) → speaker embedding; inference uses the speaker
  with most training audio by default (selectable in the UI/CLI).

## Telugu-specific

* `fastspeech2/text/telugu.py`: akshara parser producing a 53-symbol ASCII phone set (retroflexes in
  capitals: T, D, N, L, Sh) — inherent vowel *a* (no schwa deletion in Telugu), virama, all matras incl.
  the ై length-mark sequence, homorganic anusvara (ng/ny/N/n/m, word-final → m), visarga → h,
  candrabindu, gemination.
* `fastspeech2/text/numbers_te.py`: Telugu cardinals in the Indian system (వెయ్యి/వేలు/వేల, లక్ష,
  కోటి), ordinals with వ (12వ → పన్నెండవ), times (10:45 → పది నలభై ఐదు), Telugu digits ౦–౯, Latin
  acronyms spelled with Telugu letter names.
* Single-speaker selection (female voice preferred) when it covers the 6 h budget; otherwise both
  speakers with a speaker embedding.
* IndicTTS audio (48 kHz) is resampled to 22.05 kHz to match the universal HiFi-GAN.
