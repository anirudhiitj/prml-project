"""FastAPI endpoint for RCNN speech separation (Mohit's model)."""

import os
import sys
import tempfile
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

import importlib.util as _ilu

# Import model and utils from mohit's folder
_this_dir = os.path.dirname(__file__)

_model_spec = _ilu.spec_from_file_location("mohit_model", os.path.join(_this_dir, "model.py"))
_model_mod = _ilu.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_model_mod)
RCNNSeparator = _model_mod.RCNNSeparator

_utils_spec = _ilu.spec_from_file_location("mohit_utils", os.path.join(_this_dir, "utils.py"))
_utils_mod = _ilu.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils_mod)
STFTHelper = _utils_mod.STFTHelper

router = APIRouter(prefix="/mohit", tags=["RCNN"])

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pt"

_loaded_model = {}


def _ensure_checkpoint() -> Path:
    """If best_model.pt doesn't exist but .part* files do, merge them."""
    if CHECKPOINT_PATH.exists():
        return CHECKPOINT_PATH

    parts = sorted(CHECKPOINT_PATH.parent.glob(CHECKPOINT_PATH.name + ".part*"),
                   key=lambda p: int(p.suffix.lstrip(".part")))
    if not parts:
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    with open(CHECKPOINT_PATH, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
    return CHECKPOINT_PATH


def _get_model(device: str = "cpu"):
    """Load and cache the RCNN model."""
    if "model" in _loaded_model:
        return _loaded_model["model"], _loaded_model["stft"]

    ckpt_path = _ensure_checkpoint()
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})

    n_fft = model_args.get("n_fft", 512)
    hop_length = model_args.get("hop_length", 128)

    model = RCNNSeparator(
        n_fft=n_fft,
        n_sources=model_args.get("n_sources", 2),
        lstm_hidden=model_args.get("lstm_hidden", 256),
        lstm_layers=model_args.get("lstm_layers", 2),
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    stft_helper = STFTHelper(n_fft=n_fft, hop_length=hop_length)

    _loaded_model["model"] = model
    _loaded_model["stft"] = stft_helper
    return model, stft_helper


def _load_audio(path: str, target_sr: int = 8000, max_duration: float = 30.0):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    max_samples = int(target_sr * max_duration)
    if wav.shape[1] > max_samples:
        wav = wav[:, :max_samples]
    return wav.squeeze(0), target_sr


@router.post("/separate")
async def separate(
    audio: UploadFile = File(...),
    num_speakers: int = Form(2),
):
    device = "cpu"
    try:
        model, stft_helper = _get_model(device)
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    suffix = Path(audio.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        waveform, sr = _load_audio(tmp_path, target_sr=8000)
        waveform = waveform.to(device)

        # STFT
        mag, phase = stft_helper.stft(waveform)
        mag_input = mag.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)

        with torch.no_grad():
            masks = model(mag_input)  # (1, n_sources, F, T)

        output_files = []
        n_sources = min(masks.shape[1], num_speakers)
        for i in range(n_sources):
            est_mag = masks[0, i] * mag
            est_wav = stft_helper.istft(est_mag, phase, length=waveform.shape[0])
            audio_np = est_wav.cpu().numpy()
            peak = np.abs(audio_np).max()
            if peak > 0:
                audio_np = audio_np / peak * 0.95
            out_path = tmp_path + f"_speaker_{i+1}.wav"
            sf.write(out_path, audio_np, 8000, subtype="PCM_16")
            output_files.append(out_path)

        return {
            "model": "RCNN",
            "num_speakers": n_sources,
            "files": output_files,
            "sample_rate": 8000,
        }
    finally:
        os.unlink(tmp_path)
