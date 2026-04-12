"""
FastAPI endpoint for Gokul's Conv-TasNet (C++ / LibTorch).

Inference strategy (in priority order):
  1. Python model  — loads ``gokul/checkpoints/best_tasnet.pt`` via
     ``gokul/model.py`` (works on all platforms without compilation).
  2. C++ binary    — used only when the compiled ``build/inference[.exe]``
     is present AND the environment variable ``USE_CPP_BINARY=1`` is set
     (useful for Linux production servers where the binary is compiled).

Post-processing (mirrors gokul/src/inference.cpp):
  1. Polarity correction — SI-SNR loss is sign-invariant; flip each source
     whose dot-product with the mixture is negative.
  2. Global rescale — find scalar α such that α·Σ(sources) ≈ mixture in
     the least-squares sense, then scale all sources by α.
"""

from __future__ import annotations

import os
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/gokul", tags=["Conv-TasNet-CPP"])

BASE_DIR       = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
SAMPLE_RATE    = 8000
_CHUNK_SAMPLES = 32000   # 4 s @ 8 kHz (matches C++ training segment length)

# ── C++ binary path (optional) ───────────────────────────────────────────────
_INFERENCE_BIN = BASE_DIR / "build" / "inference"
if os.name == "nt":
    _cpp = BASE_DIR / "build" / "Release" / "inference.exe"
    _INFERENCE_BIN = _cpp if _cpp.exists() else BASE_DIR / "build" / "inference.exe"

# ── Lazy-loaded Python model ─────────────────────────────────────────────────
_py_model     = None   # ConvTasNet instance, loaded on first request
_py_model_err = None   # error string if loading failed


def _get_checkpoint() -> Optional[Path]:
    if not CHECKPOINT_DIR.exists():
        return None
    for f in sorted(CHECKPOINT_DIR.glob("*tasnet*.pt")):
        return f
    for f in sorted(CHECKPOINT_DIR.glob("*.pt")):
        return f
    return None


def _load_py_model():
    """Return (model, None) or (None, error_str).  Result is cached."""
    global _py_model, _py_model_err
    if _py_model is not None:
        return _py_model, None
    if _py_model_err is not None:
        return None, _py_model_err

    ckpt = _get_checkpoint()
    if ckpt is None:
        _py_model_err = "No checkpoint found in gokul/checkpoints/"
        return None, _py_model_err

    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("gokul_model_py", BASE_DIR / "model.py")
        _mod  = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        _py_model = _mod.load_model(str(ckpt), device="cpu")
        return _py_model, None
    except Exception as exc:
        _py_model_err = f"Failed to load Python model: {exc}"
        return None, _py_model_err


def _load_audio_np(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio to mono float32 at *target_sr* Hz."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    return wav


def _separate_chunked(model, wav_np: np.ndarray) -> np.ndarray:
    """
    Run Conv-TasNet on *wav_np* using non-overlapping 4-second chunks and
    return separated sources as a float32 numpy array of shape [C, T].

    Mirrors the chunked overlap-add + permutation stitching in inference.cpp.
    """
    import torch

    T = len(wav_np)
    chunk = _CHUNK_SAMPLES
    n_full    = T // chunk
    remainder = T % chunk
    n_chunks  = n_full + (1 if remainder else 0)

    chunk_results = []   # each [C, chunk]
    for i in range(n_chunks):
        start = i * chunk
        if start + chunk <= T:
            seg = wav_np[start: start + chunk]
        else:
            # Last short chunk: wrap from the beginning to fill
            have = T - start
            seg  = np.concatenate([wav_np[start:], wav_np[: chunk - have]])

        with torch.no_grad():
            x   = torch.from_numpy(seg).unsqueeze(0)   # [1, T]
            out = model(x)                              # [1, C, T]
            chunk_results.append(out.squeeze(0).numpy())  # [C, chunk]

    # ── Permutation stitching ────────────────────────────────────────────────
    # Each boundary is resolved by running a "bridge" segment centred on it.
    half = chunk // 2
    swaps = 0
    for i in range(1, n_chunks):
        boundary = i * chunk
        b_start  = boundary - half
        b_end    = boundary + half
        if b_end <= T:
            bridge_seg = wav_np[b_start: b_end]
        else:
            have = T - b_start
            bridge_seg = np.concatenate([wav_np[b_start:], wav_np[: chunk - have]])

        with torch.no_grad():
            x      = torch.from_numpy(bridge_seg).unsqueeze(0)
            bridge = model(x).squeeze(0).numpy()   # [C, chunk]

        # Second half of bridge covers same audio as first half of chunk_results[i]
        bsec   = bridge[:, half:]           # [C, half]
        cfirst = chunk_results[i][:, :half] # [C, half]

        C = bridge.shape[0]
        corr_id = sum((bsec[c] * cfirst[c]).sum() for c in range(C))
        corr_sw = sum((bsec[c] * cfirst[C - 1 - c]).sum() for c in range(C))
        if corr_sw > corr_id:
            chunk_results[i] = chunk_results[i][::-1].copy()
            swaps += 1

    # Concatenate and trim
    separated = np.concatenate(chunk_results, axis=1)[:, :T]  # [C, T]
    return separated


def _postprocess(wav_np: np.ndarray, separated: np.ndarray) -> np.ndarray:
    """
    Polarity correction + global rescale, mirroring inference.cpp post-processing.

    Args:
        wav_np:    mixture waveform  [T]
        separated: model outputs     [C, T]
    Returns:
        rescaled sources             [C, T]
    """
    C = separated.shape[0]

    # Step 1: per-source polarity correction
    for c in range(C):
        if float(np.dot(separated[c], wav_np)) < 0.0:
            separated[c] = -separated[c]

    # Step 2: global least-squares rescale so Σ sources ≈ mixture
    sum_est = separated.sum(axis=0)                          # [T]
    num     = float(np.dot(wav_np, sum_est))
    denom   = float(np.dot(sum_est, sum_est)) + 1e-8
    alpha   = num / denom
    separated = separated * alpha

    return separated


# ── Visualization ─────────────────────────────────────────────────────────────

def _generate_spectrogram(
    mixture: np.ndarray,
    sources: np.ndarray,
    save_path: str,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """
    Generate waveform + spectrogram visualization identical in style to
    Anirudh's ``generate_spectrogram_image`` in anirudh/cocktail_separation/inference.py.
    """
    n_src       = sources.shape[0]
    n_fft       = 1024
    hop_length  = 256
    n_rows      = 1 + n_src + 1   # mixture + n sources + metrics panel

    fig = plt.figure(figsize=(18, 3.2 * n_rows))
    outer_gs = gridspec.GridSpec(
        n_rows, 1,
        height_ratios=[1] * (1 + n_src) + [0.8],
        hspace=0.55,
    )

    COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]

    def _waveform(ax, wav, title, color):
        t = np.arange(len(wav)) / sample_rate
        ax.plot(t, wav, color=color, linewidth=0.6, alpha=0.85)
        ax.set_xlim(0, t[-1])
        peak = max(float(np.abs(wav).max()), 1e-6)
        ax.set_ylim(-peak * 1.1, peak * 1.1)
        ax.set_ylabel("Amplitude")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.annotate(f"peak={peak:.4f}", xy=(0.98, 0.92),
                    xycoords="axes fraction", fontsize=7, ha="right", color="gray")
        ax.tick_params(axis="x", labelsize=8)

    def _spectrogram(ax, wav):
        win = np.hanning(n_fft)
        padded = np.pad(wav, (n_fft // 2, n_fft // 2))
        frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
        S = np.abs(np.fft.rfft(frames * win))
        S_db = 20 * np.log10(np.maximum(S, 1e-10))
        t_ax = np.arange(S_db.shape[0]) * hop_length / sample_rate
        f_ax = np.arange(S_db.shape[1]) * sample_rate / n_fft
        ax.pcolormesh(t_ax, f_ax, S_db.T, shading="gouraud", cmap="magma")
        ax.set_ylabel("Freq (Hz)")
        ax.set_ylim(0, sample_rate // 2)
        ax.tick_params(axis="x", labelsize=8)

    def _add_row(row_idx, wav, label, color):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gs[row_idx],
            width_ratios=[1, 2.5], wspace=0.25,
        )
        ax_w = fig.add_subplot(inner[0])
        ax_s = fig.add_subplot(inner[1])
        _waveform(ax_w, wav, label, color)
        _spectrogram(ax_s, wav)
        ax_s.set_title(f"{label} — Spectrogram", fontsize=11, fontweight="bold")
        return ax_w, ax_s

    # Row 0: mixture
    _add_row(0, mixture, "Input Mixture", "#607D8B")

    # Rows 1…n_src: separated sources
    for i in range(n_src):
        ax_w, ax_s = _add_row(i + 1, sources[i], f"Separated — Speaker {i + 1}", COLORS[i % len(COLORS)])
        if i == n_src - 1:
            ax_w.set_xlabel("Time (s)")
            ax_s.set_xlabel("Time (s)")

    # Metrics panel
    ax_txt = fig.add_subplot(outer_gs[1 + n_src])
    ax_txt.axis("off")
    lines = ["METRICS (Conv-TasNet C++)"]
    mix_rms = float(np.sqrt(np.mean(mixture ** 2)))
    lines.append(f"  Mixture RMS: {mix_rms:.4f}")
    for i in range(n_src):
        rms  = float(np.sqrt(np.mean(sources[i] ** 2)))
        peak = float(np.abs(sources[i]).max())
        lines.append(f"  Speaker {i + 1}:  RMS={rms:.4f}  Peak={peak:.4f}")
    ax_txt.text(
        0.5, 0.5, "\n".join(lines),
        transform=ax_txt.transAxes, fontsize=10,
        fontfamily="monospace", verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="gray"),
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main separation endpoint ─────────────────────────────────────────────────

@router.post("/separate")
async def separate(
    audio: UploadFile = File(...),
    num_speakers: int = Form(2),
    model_type: str = Form("tasnet"),
):
    # Optionally use the C++ binary on compiled servers
    if os.environ.get("USE_CPP_BINARY", "0") == "1" and _INFERENCE_BIN.exists():
        return await _separate_cpp(audio, num_speakers, model_type)

    # ── Python inference path ────────────────────────────────────────────────
    model, err = _load_py_model()
    if model is None:
        return JSONResponse(status_code=503, content={"error": err})

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        wav = _load_audio_np(tmp_path, SAMPLE_RATE)

        # Peak-normalise input (mirrors C++ preprocessing)
        peak = float(np.abs(wav).max())
        if peak > 0:
            wav = wav * (0.9 / peak)

        # ── Chunked separation ───────────────────────────────────────────────
        separated = _separate_chunked(model, wav)   # [C, T]

        # ── Post-processing: polarity fix + global rescale ───────────────────
        separated = _postprocess(wav, separated)

        # ── Per-source peak normalisation before writing ─────────────────────
        stable_files: list[str] = []
        for i in range(separated.shape[0]):
            src  = separated[i].copy()
            spk  = float(np.abs(src).max())
            if spk > 0:
                src = src * (0.95 / spk)
            out_path = tmp_path + f"_source{i + 1}.wav"
            sf.write(out_path, src, SAMPLE_RATE, subtype="PCM_16")
            stable_files.append(out_path)

        # ── Visualisation ────────────────────────────────────────────────────
        viz_path = tmp_path + "_spectrogram.png"
        try:
            _generate_spectrogram(wav, separated, viz_path, SAMPLE_RATE)
        except Exception as viz_err:
            print(f"⚠️  Gokul visualization failed: {viz_err}")
            viz_path = None

        return {
            "model": f"Conv-TasNet C++ ({model_type})",
            "num_speakers": num_speakers,
            "files": stable_files,
            "sample_rate": SAMPLE_RATE,
            "spectrogram": viz_path,
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"Inference error: {exc}"},
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── C++ binary path (Linux / compiled deployments) ───────────────────────────

async def _separate_cpp(
    audio: UploadFile,
    num_speakers: int,
    model_type: str,
):
    ckpt = _get_checkpoint()
    if ckpt is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No checkpoint found in gokul/checkpoints/"},
        )

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    output_dir = tempfile.mkdtemp()
    try:
        cmd = [
            str(_INFERENCE_BIN),
            "--model",      model_type,
            "--checkpoint", str(ckpt),
            "--input",      tmp_path,
            "--output_dir", output_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={"error": f"C++ inference failed: {result.stderr}"},
            )

        stable_files: list[str] = []
        for f in sorted(Path(output_dir).glob("*.wav")):
            dst = tmp_path + f"_{f.stem}.wav"
            shutil.copy2(str(f), dst)
            stable_files.append(dst)

        return {
            "model": f"Conv-TasNet C++ (binary, {model_type})",
            "num_speakers": num_speakers,
            "files": stable_files,
            "sample_rate": SAMPLE_RATE,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        shutil.rmtree(output_dir, ignore_errors=True)
