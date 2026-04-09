"""FastAPI endpoint for Conv-TasNet (Mayank's model)."""

import sys
import os
import tempfile
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

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
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    max_samples = int(sample_rate * max_duration)
    if wav.shape[1] > max_samples:
        wav = wav[:, :max_samples]
    return wav


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

        with torch.no_grad():
            wav_in = wav.unsqueeze(0).to(device)  # (1, 1, T)
            estimates = model(wav_in)  # (1, C, T)
            estimates = estimates.squeeze(0).cpu()  # (C, T)

        output_files = []
        for i in range(estimates.shape[0]):
            audio_np = estimates[i].numpy()
            peak = np.abs(audio_np).max()
            if peak > 0:
                audio_np = audio_np / peak * 0.95
            out_path = tmp_path + f"_speaker_{i+1}.wav"
            sf.write(out_path, audio_np, 8000, subtype="PCM_16")
            output_files.append(out_path)

        return {
            "model": "Conv-TasNet",
            "num_speakers": num_speakers,
            "files": output_files,
            "sample_rate": 8000,
        }
    finally:
        os.unlink(tmp_path)
