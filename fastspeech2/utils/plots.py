"""Modified from ming024/FastSpeech2 ``utils/tools.py::plot_mel`` (adds loss curves and variance-adaptor plots).

Matplotlib figures: loss curves, variance-adaptor predictions, validation samples.

Colours follow a validated categorical order (blue, orange, aqua); one hue
per variable is kept across every figure so duration/pitch/energy are
recognisable at a glance. Phoneme labels are ASCII (ARPAbet / romanised
Telugu), so no Indic font is needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
VAR_COLOR = {"duration": BLUE, "pitch": ORANGE, "energy": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
    "font.size": 9,
})


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("step")]
    if not rows:
        return {}
    keys = [k for k in rows[0] if k]
    return {k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows]) for k in keys}


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    if len(y) < k or k <= 1:
        return y
    kernel = np.ones(k) / k
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


_PANELS = [("total", "Total loss"), ("mel", "Mel L1 (pre-PostNet)"), ("postnet", "Mel L1 (PostNet)"),
           ("duration", "Duration MSE (log)"), ("pitch", "Pitch MSE"), ("energy", "Energy MSE"),
           ("ctc", "Alignment CTC"), ("bin", "Alignment bin")]


def plot_losses(log_dir: str | Path, out_dir: str | Path, language: str) -> tuple[Path, Path]:
    log_dir, out_dir = Path(log_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va = _read_csv(log_dir / "train_log.csv"), _read_csv(log_dir / "val_log.csv")
    if not tr:
        raise FileNotFoundError(f"No training log in {log_dir}")

    def grid(title: str, series: list[tuple[dict, str, str, bool]], path: Path) -> Path:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), constrained_layout=True)
        for ax, (key, label) in zip(axes.flat, _PANELS):
            for data, name, color, smooth in series:
                if data and key in data:
                    y = data[key]
                    k = max(1, len(y) // 40) if smooth else 1
                    ax.plot(data["step"], _smooth(y, k), color=color, lw=2, label=name)
            ax.set_title(label)
            ax.set_xlabel("step")
            if key in ("total", "mel", "postnet") and np.nanmin(series[0][0].get(key, [1])) > 0:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if len(series) > 1:
            axes.flat[0].legend(loc="upper right")
        fig.suptitle(title, color=INK, fontsize=13, fontweight="semibold")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    lang = language.capitalize()
    p1 = grid(f"{lang} FastSpeech 2 — training loss (moving average)", [(tr, "train", BLUE, True)],
              out_dir / "training_loss.png")
    series = [(va, "validation", ORANGE, False)] if va else [(tr, "train", BLUE, True)]
    if va:
        series = [(va, "validation", ORANGE, False), (tr, "train (smoothed)", BLUE, True)]
    p2 = grid(f"{lang} FastSpeech 2 — validation loss (train shown for reference)", series,
              out_dir / "validation_loss.png")
    return p1, p2


def _token_axis(ax, durations: np.ndarray, frame_ms: float):
    edges = np.concatenate([[0], np.cumsum(durations)]) * frame_ms / 1000.0
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers


def _word_bands(ax, words: list[dict], edges: np.ndarray, ymax: float) -> None:
    for i, w in enumerate(words):
        if w["end"] <= w["start"]:
            continue
        x0, x1 = edges[w["start"]], edges[w["end"]]
        if i % 2 == 0:
            ax.axvspan(x0, x1, color="#f1f0ec", zorder=0, lw=0)
        label = w["word"] if w["word"].isascii() else ""
        if label and label not in ".,":
            ax.text((x0 + x1) / 2, ymax, label, ha="center", va="bottom", fontsize=7, color=INK_2, clip_on=False)


def plot_variance(result: dict, sentence_id: str, out_root: str | Path, romanized_words: list[str] | None = None) -> dict:
    """Write duration/, pitch/, energy/ and overview/ figures for one synthesised sentence."""
    out_root = Path(out_root)
    ph = result["phonemes"]
    dur = np.array(result["durations_frames"], dtype=float)
    frame_ms = result["hop_length"] / result["sampling_rate"] * 1000
    pitch = np.array(result["pitch_hz"])
    energy = np.array(result["energy"])
    words = result["words"]
    if romanized_words:
        words = [{**w, "word": r} for w, r in zip(words, romanized_words)]
    edges, centers = _token_axis(None, dur, frame_ms)
    x = np.arange(len(ph))
    width = max(10, len(ph) * 0.24)
    paths = {}

    # Duration: bars per token (ms)
    fig, ax = plt.subplots(figsize=(width, 3.8), constrained_layout=True)
    ax.bar(x, dur * frame_ms, color=BLUE, width=0.72)
    ax.set_xticks(x, ph, rotation=90, fontsize=7)
    ax.set_ylabel("predicted duration (ms)")
    ax.set_title(f"{sentence_id} — predicted phoneme durations "
                 f"(total {dur.sum() * frame_ms / 1000:.2f} s, {int(dur.sum())} frames)")
    ax.grid(axis="x", visible=False)
    _annotate_extremes(ax, x, dur * frame_ms, "{:.0f}")
    paths["duration"] = out_root / "duration" / f"{sentence_id}.png"

    def save(fig, key):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths[key], dpi=140)
        plt.close(fig)

    save(fig, "duration")

    for key, values, ylabel, fmt in (("pitch", pitch, "predicted F0 (Hz)", "{:.0f}"),
                                     ("energy", energy, "predicted energy (frame L2 norm)", "{:.1f}")):
        fig, ax = plt.subplots(figsize=(width, 3.8), constrained_layout=True)
        ax.step(edges, np.append(values, values[-1]), where="post", color=VAR_COLOR[key], lw=2)
        ax.plot(centers, values, "o", color=VAR_COLOR[key], ms=4, mec=SURFACE, mew=1)
        ymin, ymax = values.min(), values.max()
        pad = (ymax - ymin) * 0.15 + 1e-6
        ax.set_ylim(ymin - pad, ymax + pad * 2.2)
        _word_bands(ax, words, edges, ymax + pad * 1.3)
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(centers, ph, rotation=90, fontsize=6)
        sec.tick_params(colors=INK_2, length=0)
        ax.set_xlabel("time (s) — token boundaries from predicted durations")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{sentence_id} — predicted {key} per token (mean {values.mean():.1f}, range "
                     f"{ymin:.1f}–{ymax:.1f})", pad=34)
        ax.set_xlim(0, edges[-1])
        paths[key] = out_root / key / f"{sentence_id}.png"
        save(fig, key)

    # Overview: mel + three tracks on a shared time axis
    mel = np.array(result["mel"]) if "mel" in result else None
    n = 4 if mel is not None else 3
    fig, axes = plt.subplots(n, 1, figsize=(max(10, edges[-1] * 3.2), 2.2 * n), sharex=True, constrained_layout=True)
    i = 0
    if mel is not None:
        axes[0].imshow(mel.T, origin="lower", aspect="auto", extent=[0, edges[-1], 0, mel.shape[1]], cmap="magma")
        axes[0].set_ylabel("mel bin")
        axes[0].grid(False)
        axes[0].set_title(f"{sentence_id} — generated mel spectrogram and variance-adaptor predictions")
        i = 1
    for ax, (key, vals, lab) in zip(axes[i:], (("duration", dur * frame_ms, "duration (ms)"),
                                                ("pitch", pitch, "F0 (Hz)"), ("energy", energy, "energy"))):
        ax.step(edges, np.append(vals, vals[-1]), where="post", color=VAR_COLOR[key], lw=2)
        ax.set_ylabel(lab)
        for e in edges:
            ax.axvline(e, color=GRID, lw=0.4, zorder=0)
    axes[-1].set_xlabel("time (s)")
    paths["overview"] = out_root / "overview" / f"{sentence_id}.png"
    save(fig, "overview")
    return paths


def _annotate_extremes(ax, x, values, fmt: str) -> None:
    if len(values) == 0:
        return
    for idx in {int(np.argmax(values)), int(np.argmin(values))}:
        ax.annotate(fmt.format(values[idx]), (x[idx], values[idx]), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7, color=INK)


def plot_validation_sample(target: np.ndarray, pred: np.ndarray, attn: np.ndarray, path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for ax, data, name in ((axes[0], target.T, "target mel"), (axes[1], pred.T, "predicted mel (teacher-forced)")):
        ax.imshow(data, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(name)
        ax.grid(False)
    axes[2].imshow(attn.T, origin="lower", aspect="auto", cmap="Blues")
    axes[2].set_title("learned soft alignment (text × mel frames)")
    axes[2].set_xlabel("mel frame")
    axes[2].set_ylabel("token")
    axes[2].grid(False)
    fig.suptitle(title, color=INK)
    fig.savefig(path, dpi=110)
    plt.close(fig)
