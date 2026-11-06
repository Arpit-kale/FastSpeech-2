"""NEW FILE (not in ming024/FastSpeech2): PDF/Markdown report builder.

Build ``Report/Technical_Report.pdf`` (and a Markdown twin) from configs, logs and results.

One content model (a list of blocks) is rendered twice: to PDF with
ReportLab and to Markdown (which keeps Telugu script, since ReportLab cannot
shape Indic text — the PDF shows Telugu sentences in romanised form).
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..utils import get_logger
from .analysis import describe_losses, describe_sentence, load_language_artifacts, sentence_stats

log = get_logger("report")
TELUGU = re.compile(r"[ఀ-౿]")


# ---------------------------------------------------------------- content model
@dataclass
class Block:
    kind: str  # h1 h2 h3 p bullets table image pagebreak
    text: str = ""
    items: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    path: Path | None = None
    width: float = 1.0
    alt_text: str = ""  # PDF-safe variant of text (romanised)


class Doc:
    def __init__(self):
        self.blocks: list[Block] = []

    def h1(self, t):
        self.blocks.append(Block("h1", t))

    def h2(self, t):
        self.blocks.append(Block("h2", t))

    def h3(self, t):
        self.blocks.append(Block("h3", t))

    def p(self, t, alt=""):
        self.blocks.append(Block("p", t, alt_text=alt))

    def bullets(self, items):
        self.blocks.append(Block("bullets", items=list(items)))

    def table(self, rows):
        self.blocks.append(Block("table", rows=rows))

    def image(self, path, width=1.0, caption=""):
        if path and Path(path).exists():
            self.blocks.append(Block("image", text=caption, path=Path(path), width=width))

    def pagebreak(self):
        self.blocks.append(Block("pagebreak"))


# ---------------------------------------------------------------- static text
ARCH = [
    "**Encoder** — phoneme embedding + sinusoidal positions + 4 Feed-Forward Transformer (FFT) blocks "
    "(256-d, 2 heads, Conv1D FFN 1024 with kernels [9, 1], dropout 0.2).",
    "**Variance adaptor** — duration, pitch and energy predictors (2 × [Conv1D(k=3) → ReLU → LayerNorm → "
    "Dropout] → Linear). Pitch and energy are predicted at phoneme level, quantised into 256 bins and added "
    "back as learned embeddings; the length regulator expands each phoneme by its duration.",
    "**Decoder** — 6 FFT blocks on the expanded sequence, a linear projection to 80 mel bins and a 5-layer "
    "PostNet (512 channels) that predicts a residual refinement.",
    "**Vocoder** — the universal HiFi-GAN V1 generator released with the reference implementation "
    "(22.05 kHz, hop 256, 80 mels, fmax 8 kHz); Griffin-Lim is used automatically if it is unavailable.",
    "**Losses** — L1 on mel before and after PostNet, MSE on log-duration, phoneme-level pitch and energy.",
]

MODIFICATIONS = [
    ["Area", "Reference (ming024)", "This project", "Why"],
    ["Durations", "Montreal Forced Aligner TextGrids", "Learned aligner (RAD-TTS: conv encoders, "
     "beta-binomial prior, CTC forward-sum loss, Monotonic Alignment Search, binarisation loss)",
     "MFA has no Telugu model/lexicon; Svarah accents fit US-English models poorly; fully automatic"],
    ["Pitch / energy targets", "Averaged with MFA durations at preprocessing", "Frame-level features stored; "
     "phoneme averages computed on the fly from the MAS alignment", "Consistent with learned durations"],
    ["Pitch normalisation", "Global mean/std", "Per-speaker mean/std (1-99 percentile robust), "
     "unvoiced frames interpolated", "Multi-speaker Svarah; male/female F0 ranges differ"],
    ["English text", "g2p_en, basic cleaners", "Normaliser for times (10:45 AM), dates, years, ordinals, "
     "currency; g2p_en on the whole sentence (POS-aware homographs)", "E2 homograph and E4 normalisation"],
    ["Telugu text", "Not supported", "Unicode akshara parser -> 53-symbol phone set (inherent vowel, virama, "
     "matras, homorganic anusvara, visarga, gemination); Telugu number/ordinal/time expansion", "Telugu support"],
    ["Batching", "Fixed batch size", "Frame-budget bucketed batches", "Variable-length Telugu clips; fewer OOMs"],
    ["Training", "Fixed schedule", "AdamW + Noam warm-up, AMP fp16, grad clipping, resume, best-by-val "
     "checkpoint, CSV/TensorBoard logs, divergence guard", "Robust, reproducible runs"],
    ["Checkpoint", "Weights + optimiser", "Self-contained fp16 inference checkpoint (vocab, speakers, stats, "
     "config) ≈ 70 MB", "Fits GitHub's 100 MB limit; one file to deploy"],
    ["Attention", "Custom MHA", "torch.nn.MultiheadAttention (batch-first, key padding mask)", "Faster, fused kernels"],
]

LIMITATIONS = [
    "**Data size.** 5 h is roughly a fifth of LJSpeech; prosody predictors (pitch/duration) generalise less "
    "well, so long or unusual sentences (E1, T1) tend to have flatter F0 and less natural rhythm.",
    "**Svarah is multi-speaker spontaneous + read ASR data** (16 kHz, varied microphones, ~5 min per speaker). "
    "A speaker embedding helps, but accent/channel variation per speaker and band-limited audio limit "
    "English quality relative to the studio-recorded Telugu IndicTTS data.",
    "**Homographs.** English relies on POS tagging (works for 'conduct'); Telugu 'పట్టు' has the same "
    "pronunciation in both senses, so the frontend cannot—and need not—disambiguate it.",
    "**Malformed input** ('rhyth m', 'అడ విలో') is passed through as-is: English spells the stray letter "
    "('em'); Telugu happens to recover the same phones. A spell-correction/word-joining pass would fix E3.",
    "**Code-mixing** ('andaroo') falls back to g2p_en's neural OOV model trained on English spellings.",
    "**Vocoder mismatch.** The universal HiFi-GAN was not fine-tuned on predicted mels of these speakers; "
    "fine-tuning it on ground-truth-aligned outputs would reduce metallic artefacts.",
]

IMPROVEMENTS = [
    "Fine-tune HiFi-GAN on each model's teacher-forced mel outputs.",
    "Train longer or use more data (e.g. all IndicTTS Telugu for one speaker; full Svarah), or pre-train on "
    "LJSpeech/IndicTTS and fine-tune on the 5 h subset.",
    "Add a text pre-processor that rejoins split words (E3/T3) and a Telugu loanword lexicon (/f/ for ఫ).",
    "Replace the pitch predictor with a CWT/frame-level predictor (paper's variant) for more expressive F0.",
    "Use the phoneme-level word-boundary token and punctuation-aware pauses to improve phrasing in long sentences.",
]


def _todo(v: str | None) -> bool:
    return not v or str(v).strip().upper().startswith("TODO")


def _romanize(res: dict) -> str:
    if not res:
        return ""
    words = []
    for w in res["words"]:
        ph = res["phonemes"][w["start"]:w["end"]]
        if ph in (["sp"], ["sil"]):
            words[-1:] = [words[-1] + ("," if ph == ["sp"] else ".")] if words else []
        else:
            words.append("".join(ph))
    return " ".join(words)


def _cfg_rows(cfg) -> list[list[str]]:
    d, a, m, t, o = cfg.dataset, cfg.audio, cfg.model, cfg.train, cfg.optimizer
    return [
        ["Dataset / split", f"{d.hf_name} [{d.hf_split}]"],
        ["Target hours (train / val)", f"{d.train_hours} / {d.val_hours}"],
        ["Clip length filter", f"{d.min_duration}–{d.max_duration} s (ceiling {d.get('max_duration_ceiling')})"],
        ["Speaker policy", {True: "single speaker", False: "all speakers (speaker embedding)",
                            "auto": f"auto (prefer '{d.get('preferred_speaker')}' if it covers the budget)"}
         .get(d.get("single_speaker"), str(d.get("single_speaker")))],
        ["Audio", f"{a.sampling_rate} Hz, FFT {a.n_fft}, hop {a.hop_length}, {a.n_mels} mels, fmax {a.mel_fmax}"],
        ["F0 range (WORLD)", f"{a.f0_min}–{a.f0_max} Hz"],
        ["Encoder / decoder", f"{m.encoder.layers} / {m.decoder.layers} FFT blocks, d={m.encoder.hidden}"],
        ["Optimiser", f"AdamW peak lr {o.peak_lr}, Noam warm-up {o.warmup_steps}, betas {o.betas}"],
        ["Batching", f"≤{t.max_frames_per_batch} frames, ≤{t.max_batch_size} utts, fp16={t.fp16}"],
        ["Steps / validation", f"{t.total_steps} steps, validate every {t.val_every}"],
        ["Aligner bin-loss ramp", f"start {m.aligner.bin_loss_start_step}, warm-up {m.aligner.bin_loss_warmup_steps}"],
    ]


def _selection_rows(art: dict) -> list[list[str]]:
    sel, summ = art["selection"], art["summary"]
    if not sel:
        return [["Status", "Dataset not yet prepared (run preprocess.py)"]]
    rows = [
        ["Rows in source split", str(sel.get("candidates_total"))],
        ["After filters + de-duplication", str(sel.get("candidates_after_filters"))],
        ["Max clip duration used", f"{sel.get('max_duration_used')} s"],
        ["Speakers used", f"{len(sel.get('speakers_used', []))} ({'single' if sel.get('single_speaker') else 'multi'}-speaker)"],
        ["Train utterances / hours", f"{sel.get('train_utterances')} / {sel.get('train_hours_after_trim')} h (before trim {sel.get('train_hours_before_trim')} h)"],
        ["Val utterances / hours", f"{sel.get('val_utterances')} / {sel.get('val_hours_after_trim')} h (before trim {sel.get('val_hours_before_trim')} h)"],
    ]
    if summ:
        s = summ["splits"]
        rows.append(["After feature extraction", f"train {s.get('train', {}).get('utterances')} utt, "
                                                  f"val {s.get('val', {}).get('utterances')} utt"])
        rows.append(["Default inference speaker", str(summ.get("default_speaker"))])
    return rows


def build_content(cfgs, obs: dict, sentences: dict) -> Doc:
    doc = Doc()
    langs = [(c, load_language_artifacts(c)) for c in cfgs]
    author = obs.get("author", "")
    doc.h1("Implementation of FastSpeech 2-Based Text-to-Speech Synthesis for English and Telugu")
    doc.p(f"Technical report — {'' if _todo(author) else author + ' — '}{dt.date.today():%d %B %Y}"
          + ("" if _todo(obs.get("repo_url")) else f" — {obs['repo_url']}"))

    doc.h2("1. Objective and overview")
    doc.p("Two FastSpeech 2 models were implemented and trained independently: an English model on a ~5 h "
          "subset of Svarah (Indian-accented English) and a Telugu model on a ~5 h subset of IndicTTS Telugu, "
          "each validated on ~1 h of held-out audio. Starting from ming024/FastSpeech2, the code was "
          "restructured into a reusable package with language-specific frontends and a learned aligner, and "
          "the whole workflow (download → selection → features → training → inference → plots → this "
          "report) is automated. A FastAPI web application serves both models for interactive synthesis.")

    doc.h2("2. Implementation approach")
    doc.h3("2.1 Architecture components used from the reference implementation")
    doc.bullets(ARCH)
    doc.h3("2.2 Modifications for English and Telugu")
    doc.table(MODIFICATIONS)
    doc.p("Telugu G2P example: సమావేశం → s a m aa v ee sh a m (word-final anusvara → m); బలంగా → b a l a ng g aa "
          "(anusvara assimilates to the velar); పట్టు → p a T T u (gemination via virama; capitals mark retroflexes).",
          alt="Telugu G2P example: samaaveesham -> s a m aa v ee sh a m (word-final anusvara -> m); balangaa -> "
              "b a l a ng g aa (anusvara assimilates to the velar); paTTu -> p a T T u (gemination; capitals = retroflex).")

    doc.h2("3. Datasets, subsampling and preprocessing")
    doc.p("Both corpora are processed by the same deterministic procedure (seed 1234): (1) read metadata only; "
          "(2) keep clips of 1–15 s whose text passes a script check (≥60 % Latin letters for English, ≥70 % "
          "Telugu characters for Telugu), relaxing the upper limit in 2 s steps only if the pool is too small; "
          "(3) de-duplicate on normalised text so the same prompt can never be in both splits; (4) choose "
          "speakers (Telugu: one speaker if it alone covers 6 h — the female voice is preferred — otherwise "
          "both; English: all speakers, as Svarah has only minutes per speaker); (5) draw validation first, "
          "round-robin across speakers (stratified), then training from the remainder. Disjointness of clip "
          "indices and of texts is asserted in code and covered by a unit test.")
    doc.p("Preprocessing: decode → resample to 22.05 kHz (soxr) → trim leading/trailing silence (35 dB) → "
          "peak-normalise → 80-bin log-mel (HiFi-GAN recipe) → WORLD DIO+StoneMask F0 on the same hop → "
          "frame energy (STFT L2 norm). F0 is interpolated through unvoiced regions and z-normalised per "
          "speaker; energy is z-normalised globally. Text is normalised and converted to phones; utterances "
          "with fewer mel frames than tokens are dropped (the aligner needs ≥1 frame per token).")
    for cfg, art in langs:
        doc.h3(f"3.{1 + langs.index((cfg, art))} {cfg.language.capitalize()} — {cfg.dataset.hf_name}")
        doc.table([["Item", "Value"], *_selection_rows(art)])

    doc.h2("4. Training configuration")
    for cfg, art in langs:
        doc.h3(f"{cfg.language.capitalize()}")
        doc.table([["Setting", "Value"], *_cfg_rows(cfg)])

    doc.pagebreak()
    doc.h2("5. Training and validation loss analysis")
    for cfg, art in langs:
        name = cfg.language.capitalize()
        doc.h3(name)
        doc.image(art["results_dir"] / "training_loss.png", 1.0, f"{name}: training loss components")
        doc.image(art["results_dir"] / "validation_loss.png", 1.0, f"{name}: validation loss (train overlaid)")
        for para in describe_losses(name, art["losses"]):
            doc.p(para)

    doc.pagebreak()
    doc.h2("6. Generated speech and variance-adaptor analysis")
    doc.p("Each test sentence was synthesised with the best-validation checkpoint and the default speaker at "
          "neutral controls (duration/pitch/energy = 1.0). For each sentence the duration (ms per token), "
          "pitch (Hz per token, denormalised with the speaker's statistics) and energy predictions are plotted "
          "on a shared time axis with word spans shaded. The quantitative notes below are computed from the "
          "predictions; the listening notes are the author's observations of the audio.")
    for cfg, art in langs:
        lang = cfg.language
        doc.h3(f"6.{1 + langs.index((cfg, art))} {lang.capitalize()}")
        by_id = {r["id"]: r for r in art["results"]}
        for item in sentences[lang]:
            sid = item["id"]
            res = by_id.get(sid)
            text = item["text"]
            doc.p(f"**{sid}** — {text}", alt=f"**{sid}** — {_romanize(res)} (romanised; original Telugu script in "
                  "data/test_sentences.json)" if TELUGU.search(text) and res else "")
            doc.p(f"*Focus:* {item['focus']}")
            if not res:
                doc.p("*Not synthesised yet — run inference.py after training.*")
                continue
            st = sentence_stats(res)
            doc.bullets(describe_sentence(sid, res, st, lang))
            note = obs.get("listening", {}).get(sid)
            doc.p(f"*Listening notes:* {note}" if not _todo(note) else
                  "*Listening notes: pending — fill Report/observations.yaml after listening.*")
            plots = art["results_dir"] / "variance_adaptor_plots"
            for kind in ("duration", "pitch", "energy"):
                doc.image(plots / kind / f"{sid}.png", 1.0, f"{sid}: predicted {kind}")
            word_rows = [["Word", "Phones", "ms", "F0 Hz", "Energy"]] + [
                [w["word"] if not TELUGU.search(w["word"]) else w["phones"].replace(" ", ""), w["phones"],
                 str(w["ms"]), f"{w['f0']:.0f}", f"{w['energy']:.2f}"] for w in st["words"] if w["phones"] not in ("sp", "sil")
            ]
            doc.table(word_rows)
        overall = obs.get("overall", {}).get(lang)
        if not _todo(overall):
            doc.p(f"*Overall listening impression ({lang}):* {overall}")
        doc.pagebreak()

    doc.h2("7. English vs Telugu: comparison, limitations and improvements")
    comp = [["Metric", *[c.language.capitalize() for c, _ in langs]]]

    def metric(fn):
        vals = []
        for _, art in langs:
            try:
                vals.append(fn(art))
            except Exception:
                vals.append("—")
        return vals

    stats_all = {c.language: [sentence_stats(r) for r in a["results"]] for c, a in langs}
    comp += [
        ["Best validation loss (step)", *metric(lambda a: f"{a['losses']['best_val_total']:.3f} ({a['losses']['best_val_step']})")],
        ["Final val mel L1 (PostNet)", *metric(lambda a: f"{a['losses']['final_val_components']['postnet']:.3f}")],
        ["Final val duration / pitch MSE", *metric(lambda a: f"{a['losses']['final_val_components']['duration']:.3f} / {a['losses']['final_val_components']['pitch']:.3f}")],
        ["Over-fitting detected", *metric(lambda a: "yes" if a["losses"]["overfitting"] else "no")],
        ["Speaking rate on tests (phones/s)", *[f"{sum(s['phones_per_s'] for s in v) / len(v):.1f}" if v else "—" for v in stats_all.values()]],
        ["Mean predicted F0 on tests (Hz)", *[f"{sum(s['pitch_mean'] for s in v) / len(v):.0f}" if v else "—" for v in stats_all.values()]],
        ["Training speakers", *metric(lambda a: str(len(a["selection"]["speakers_used"])))],
    ]
    doc.table(comp)
    doc.p("Telugu training data is single-speaker studio speech with a consistent reading style, whereas Svarah "
          "mixes many speakers, accents, microphones and spontaneous speech at 16 kHz. With an equal 5 h budget "
          "this generally gives Telugu lower mel and pitch losses and steadier prosody, while the English model "
          "must average over speakers. Telugu's near-phonemic orthography also makes the rule-based G2P almost "
          "error-free, whereas English pronunciation depends on CMUdict/neural G2P and POS tagging. Conversely, "
          "Telugu words are long and agglutinative (T1, T4 'ప్రారంభమవుతుంది'), so sequences are longer and duration "
          "errors accumulate over more tokens.",
          alt="Telugu training data is single-speaker studio speech with a consistent reading style, whereas Svarah "
          "mixes many speakers, accents, microphones and spontaneous speech at 16 kHz. With an equal 5 h budget "
          "this generally gives Telugu lower mel and pitch losses and steadier prosody, while the English model "
          "must average over speakers. Telugu's near-phonemic orthography also makes the rule-based G2P almost "
          "error-free, whereas English pronunciation depends on CMUdict/neural G2P and POS tagging. Conversely, "
          "Telugu words are long and agglutinative (T1, T4 'praarambhamavutundi'), so sequences are longer and "
          "duration errors accumulate over more tokens.")
    doc.h3("Limitations and failure cases")
    doc.bullets(LIMITATIONS)
    doc.h3("Possible improvements")
    doc.bullets(IMPROVEMENTS)

    doc.h2("8. Reproducibility")
    doc.table([
        ["Step", "Command"],
        ["Install", "pip install -r requirements.txt"],
        ["Svarah access", "accept terms on huggingface.co/datasets/ai4bharat/Svarah; export HF_TOKEN=..."],
        ["Everything, both languages", "bash scripts/run_all.sh   (or: make all)"],
        ["Preprocess", "python English/preprocess.py ; python Telugu/preprocess.py"],
        ["Train (+ loss plots)", "python English/train.py ; python Telugu/train.py"],
        ["Inference (+ variance plots)", "python English/inference.py ; python Telugu/inference.py"],
        ["Report", "python -m fastspeech2.cli report --config English/config.yaml"],
        ["Web app", "uvicorn app.server:app --port 7860   (or: docker compose up)"],
        ["Tests", "pytest -q"],
    ])
    doc.p("Configurations: English/config.yaml and Telugu/config.yaml (every value can be overridden with "
          "--set key=value). Seeds are fixed (1234) for data selection and training; the selection report "
          "(data/<lang>/raw/selection_report.json) records exactly which clips were used. Software: Python ≥3.10, "
          "PyTorch ≥2.1, librosa, pyworld, numba, g2p_en, datasets — pinned in requirements.txt.")
    doc.h2("References")
    doc.bullets([
        "Y. Ren et al., FastSpeech 2: Fast and High-Quality End-to-End Text to Speech, ICLR 2021 (arXiv:2006.04558).",
        "C.-M. Chien, ming024/FastSpeech2 (GitHub), reference implementation.",
        "R. Badlani et al., One TTS Alignment To Rule Them All, ICASSP 2022.",
        "J. Kong et al., HiFi-GAN, NeurIPS 2020.",
        "T. Javed et al., Svarah: Evaluating English ASR Systems on Indian Accents, Interspeech 2023.",
        "IndicTTS (IIT Madras) Telugu corpus, via SPRINGLab/IndicTTS_Telugu on Hugging Face.",
    ])
    return doc


# ---------------------------------------------------------------- renderers
def _md_inline(t: str) -> str:
    return t


def render_markdown(doc: Doc, out: Path, rel_to: Path) -> Path:
    lines = []
    for b in doc.blocks:
        if b.kind == "h1":
            lines += [f"# {b.text}", ""]
        elif b.kind == "h2":
            lines += [f"## {b.text}", ""]
        elif b.kind == "h3":
            lines += [f"### {b.text}", ""]
        elif b.kind == "p":
            lines += [b.text, ""]
        elif b.kind == "bullets":
            lines += [f"- {i}" for i in b.items] + [""]
        elif b.kind == "table":
            head, *rest = b.rows
            lines.append("| " + " | ".join(str(c).replace("|", "/") for c in head) + " |")
            lines.append("|" + "---|" * len(head))
            lines += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rest]
            lines.append("")
        elif b.kind == "image":
            rel = Path("..") / b.path.relative_to(rel_to) if b.path.is_relative_to(rel_to) else b.path
            lines += [f"![{b.text}]({rel.as_posix()})", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _pdf_text(t: str) -> str:
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    t = t.replace("≥", "&gt;=").replace("≤", "&lt;=").replace("≈", "~").replace("→", "-&gt;")
    if TELUGU.search(t):  # ReportLab cannot shape Telugu: transliterate with the G2P
        from ..text.telugu import word_to_phones

        t = re.sub(r"[\u0C00-\u0C7F]+", lambda m: "".join(word_to_phones(m.group(0))), t)
    return t


def render_pdf(doc: Doc, out: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    ss = getSampleStyleSheet()
    ink, accent, rule = colors.HexColor("#1b1b1a"), colors.HexColor("#2a78d6"), colors.HexColor("#e2e1dc")
    st = {
        "h1": ParagraphStyle("h1", parent=ss["Title"], fontSize=19, leading=24, textColor=ink, alignment=TA_LEFT, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=14, textColor=accent, spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontSize=11.5, textColor=ink, spaceBefore=8, spaceAfter=4),
        "p": ParagraphStyle("p", parent=ss["BodyText"], fontSize=9.5, leading=13.5, textColor=ink, spaceAfter=5),
        "cell": ParagraphStyle("cell", parent=ss["BodyText"], fontSize=7.8, leading=10, textColor=ink),
        "cap": ParagraphStyle("cap", parent=ss["BodyText"], fontSize=8, textColor=colors.HexColor("#6b6a66"), spaceAfter=8),
    }
    page_w = A4[0] - 3.6 * cm
    story = []
    for b in doc.blocks:
        if b.kind in ("h1", "h2", "h3"):
            story.append(Paragraph(_pdf_text(b.text), st[b.kind]))
        elif b.kind == "p":
            txt = b.alt_text or b.text
            if txt.strip():
                story.append(Paragraph(_pdf_text(txt), st["p"]))
        elif b.kind == "bullets":
            items = [ListItem(Paragraph(_pdf_text(i), st["p"]), leftIndent=10) for i in b.items]
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8))
        elif b.kind == "table":
            data = [[Paragraph(_pdf_text(str(c)), st["cell"]) for c in r] for r in b.rows]
            n = len(b.rows[0])
            widths = [page_w / n] * n if n != 2 else [page_w * 0.32, page_w * 0.68]
            if n == 4:
                widths = [page_w * w for w in (0.14, 0.22, 0.40, 0.24)]
            if n == 5:
                widths = [page_w * w for w in (0.22, 0.38, 0.12, 0.14, 0.14)]
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f0ec")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, rule),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story += [t, Spacer(1, 6)]
        elif b.kind == "image":
            from reportlab.lib.utils import ImageReader

            iw, ih = ImageReader(str(b.path)).getSize()
            w = page_w * b.width
            h = w * ih / iw
            if h > 20 * cm:
                h = 20 * cm
                w = h * iw / ih
            story += [Image(str(b.path), width=w, height=h), Paragraph(_pdf_text(b.text), st["cap"])]
        elif b.kind == "pagebreak":
            story.append(PageBreak())

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6b6a66"))
        canvas.drawString(1.8 * cm, 1.1 * cm, "FastSpeech 2 — English & Telugu — Technical Report")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {d.page}")
        canvas.restoreState()

    out.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                      bottomMargin=1.8 * cm, title="FastSpeech 2 Technical Report").build(
        story, onFirstPage=footer, onLaterPages=footer)
    return out


def build_report(cfgs, report_dir: str | Path) -> Path:
    import json

    report_dir = Path(report_dir)
    root = Path(cfgs[0].project_root)
    obs_path = report_dir / "observations.yaml"
    obs = yaml.safe_load(obs_path.read_text(encoding="utf-8")) if obs_path.exists() else {}
    sentences = json.loads((root / "data" / "test_sentences.json").read_text(encoding="utf-8"))
    doc = build_content(cfgs, obs or {}, sentences)
    md = render_markdown(doc, report_dir / "Technical_Report.md", root)
    pdf = render_pdf(doc, report_dir / "Technical_Report.pdf")
    log.info("Report written: %s and %s", pdf, md)
    _ = shutil  # keep import for potential asset copying
    return pdf
