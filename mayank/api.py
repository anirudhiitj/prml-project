"""FastAPI endpoint for Conv-TasNet (Mayank's model)."""

import sys
import os
import tempfile
import torch
import torchaudio
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse


# --- Asteroid import check ---
try:
    import asteroid
except ImportError as e:
    raise ImportError("\n\n[Mayank model] Asteroid library is not installed.\n\nTo use this model, run:\n  pip install asteroid --no-deps\n  pip install asteroid-filterbanks julius pytorch-lightning 'torch-optimizer<0.2.0' 'torchmetrics<=0.11.4'\n\nSee Global_How_To_Run.md for details.\n") from e

import importlib.util as _ilu
_model_spec = _ilu.spec_from_file_location("mayank_model", os.path.join(os.path.dirname(__file__), "model.py"))
_model_mod = _ilu.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_model_mod)
SpeechSeparator = _model_mod.SpeechSeparator

router = APIRouter(prefix="/mayank", tags=["Conv-TasNet"])

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

_loaded_models = {}


def _get_model(n_src: int, device: str = "cpu"):
    """Load and cache the Conv-TasNet model."""
    # Look for any .pt file in checkpoints/
    ckpt_path = None
    if CHECKPOINT_DIR.exists():
        for f in sorted(CHECKPOINT_DIR.glob("*.pt")):
            ckpt_path = f
            break

    key = f"{n_src}spk"
    if key not in _loaded_models:
        model = SpeechSeparator(
            n_src=n_src, sample_rate=8000,
            n_blocks=8, n_repeats=3, bn_chan=128, hid_chan=512,
        )
        if ckpt_path and ckpt_path.exists():
            state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
        model.eval()
        _loaded_models[key] = model

    return _loaded_models[key]


def _load_audio(path: str, sample_rate: int = 8000, max_duration: float = 30.0):
    wav, sr = sf.read(path, dtype="float32")
    # Convert to mono if stereo
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # Resample if needed
    if sr != sample_rate:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sample_rate, sr)
        wav = resample_poly(wav, sample_rate // g, sr // g).astype(np.float32)
    # Truncate
    max_samples = int(sample_rate * max_duration)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    return torch.from_numpy(wav).unsqueeze(0)  # (1, T)


def _extract_intermediates(model, wav_input, device):
    """
    Run inference while capturing encoder output and speaker masks
    from the internal Conv-TasNet layers.
    """
    model.eval()
    with torch.no_grad():
        x = wav_input.clone()
        if x.dim() == 2:
            x = x.unsqueeze(0)

        conv_tasnet = model.model

        encoder_output = conv_tasnet.forward_encoder(x)
        est_masks = conv_tasnet.forward_masker(encoder_output)
        masked_tf_rep = conv_tasnet.apply_masks(encoder_output, est_masks)
        decoded = conv_tasnet.forward_decoder(masked_tf_rep)

    return (
        encoder_output.squeeze(0).cpu().numpy(),
        est_masks.squeeze(0).cpu().numpy(),
        decoded.squeeze(0).cpu().numpy(),
    )


def _generate_visualization(mix_np, encoder_output, masks, separated,
                            sample_rate, n_src, save_path):
    """Generate a paper-style Conv-TasNet internal analysis plot."""

    n_panels = 3 + n_src  # mixture + encoder + n_src masks + separated
    fig = plt.figure(figsize=(20, 4 * (n_panels + 1)), dpi=150)
    fig.patch.set_facecolor('#0a0a1a')

    gs = gridspec.GridSpec(n_panels + 1, 1, hspace=0.35,
                           left=0.06, right=0.97, top=0.95, bottom=0.03)

    time_wav = np.arange(len(mix_np)) / sample_rate
    n_frames = encoder_output.shape[1]

    enc_display = np.abs(encoder_output)
    sort_idx = np.argsort(enc_display.mean(axis=1))
    enc_sorted = enc_display[sort_idx]

    speaker_colors = ['#00d4ff', '#ff6b6b', '#50fa7b', '#bd93f9']

    def style_axis(ax):
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#888888')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#333333')

    # Panel 1: Mixture Waveform
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_wav, mix_np, color='#ffffff', linewidth=0.3, alpha=0.9)
    ax1.fill_between(time_wav, mix_np, alpha=0.15, color='#00d4ff')
    ax1.set_ylabel('Amplitude', fontsize=11, color='white', fontweight='bold')
    ax1.set_title('\u2460 Input Mixture Waveform', fontsize=14, color='#00d4ff',
                  fontweight='bold', pad=10, loc='left')
    ax1.set_xlim(0, time_wav[-1])
    style_axis(ax1)

    # Panel 2: Encoder Output
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(enc_sorted, aspect='auto', cmap='inferno',
                     extent=[0, time_wav[-1], 0, enc_sorted.shape[0]],
                     interpolation='bilinear', origin='lower')
    ax2.set_ylabel('Filter Index', fontsize=11, color='white', fontweight='bold')
    ax2.set_title('\u2461 Encoder Output (512 Learned Filters \u00d7 Time)',
                  fontsize=14, color='#ff9f43', fontweight='bold', pad=10, loc='left')
    style_axis(ax2)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.015, pad=0.01)
    cbar2.ax.tick_params(colors='#888888', labelsize=8)

    # Panels 3+: Speaker Masks
    for s in range(n_src):
        ax_m = fig.add_subplot(gs[2 + s])
        mask_sorted = masks[s][sort_idx]
        im_m = ax_m.imshow(mask_sorted, aspect='auto', cmap='magma',
                           extent=[0, time_wav[-1], 0, mask_sorted.shape[0]],
                           interpolation='bilinear', vmin=0, vmax=1, origin='lower')
        ax_m.set_ylabel('Filter Index', fontsize=11, color='white', fontweight='bold')
        ax_m.set_title(f'\u2462 Speaker {s+1} Mask  (0 = suppress, 1 = keep)',
                       fontsize=14, color=speaker_colors[s % len(speaker_colors)],
                       fontweight='bold', pad=10, loc='left')
        style_axis(ax_m)
        cbar_m = plt.colorbar(im_m, ax=ax_m, fraction=0.015, pad=0.01)
        cbar_m.ax.tick_params(colors='#888888', labelsize=8)

    # Final Panel: Separated Waveforms
    ax_sep = fig.add_subplot(gs[2 + n_src])
    for s in range(n_src):
        t_sep = np.arange(len(separated[s])) / sample_rate
        ax_sep.plot(t_sep, separated[s], color=speaker_colors[s % len(speaker_colors)],
                    linewidth=0.4, alpha=0.85, label=f'Speaker {s+1}')
    ax_sep.set_ylabel('Amplitude', fontsize=11, color='white', fontweight='bold')
    ax_sep.set_xlabel('Time (seconds)', fontsize=11, color='white', fontweight='bold')
    ax_sep.set_title('\u2463 Separated Speaker Waveforms', fontsize=14,
                     color='#50fa7b', fontweight='bold', pad=10, loc='left')
    ax_sep.set_xlim(0, time_wav[-1])
    style_axis(ax_sep)
    ax_sep.legend(loc='upper right', fontsize=10, facecolor='#1a1a2e',
                  edgecolor='#333333', labelcolor='white')

    fig.suptitle('Conv-TasNet Internal Analysis',
                 fontsize=18, color='white', fontweight='bold', y=0.98)

    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


@router.post("/separate")
async def separate(
    audio: UploadFile = File(...),
    num_speakers: int = Form(2),
):
    device = "cpu"
    model = _get_model(num_speakers, device)

    suffix = Path(audio.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        wav = _load_audio(tmp_path, sample_rate=8000, max_duration=30.0)
        mix_np = wav.squeeze().numpy().copy()

        wav_in = wav.unsqueeze(0).to(device)  # (1, 1, T)
        encoder_output, masks, separated = _extract_intermediates(model, wav_in, device)

        # separated is (n_src, T) numpy array
        output_files = []
        for i in range(separated.shape[0]):
            audio_np = separated[i]
            peak = np.abs(audio_np).max()
            if peak > 0:
                audio_np = audio_np / peak * 0.95
            out_path = tmp_path + f"_speaker_{i+1}.wav"
            sf.write(out_path, audio_np, 8000, subtype="PCM_16")
            output_files.append(out_path)

        # Generate visualization
        viz_path = tmp_path + "_analysis.png"
        _generate_visualization(
            mix_np, encoder_output, masks, separated,
            8000, num_speakers, viz_path
        )

        return {
            "model": "Conv-TasNet",
            "num_speakers": num_speakers,
            "files": output_files,
            "sample_rate": 8000,
            "visualization": viz_path,
        }
    finally:
        os.unlink(tmp_path)
