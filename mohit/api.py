"""FastAPI endpoint for RCNN Separator (Mohit's model)."""

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
_base = os.path.dirname(__file__)
_model_spec = _ilu.spec_from_file_location("mohit_model", os.path.join(_base, "model.py"))
_model_mod = _ilu.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_model_mod)
RCNNSeparator = _model_mod.RCNNSeparator

_utils_spec = _ilu.spec_from_file_location("mohit_utils", os.path.join(_base, "utils.py"))
_utils_mod = _ilu.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils_mod)
STFTHelper = _utils_mod.STFTHelper
normalize_waveform = _utils_mod.normalize_waveform

router = APIRouter(prefix="/mohit", tags=["RCNN"])

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

_loaded_models = {}


def _get_model(device: str = "cpu"):
    """Load and cache the RCNN model."""
    if "rcnn" in _loaded_models:
        return _loaded_models["rcnn"], _loaded_models["rcnn_meta"]

    ckpt_path = None
    if CHECKPOINT_DIR.exists():
        for f in sorted(CHECKPOINT_DIR.glob("*.pt")):
            ckpt_path = f
            break

    if ckpt_path is None:
        # No checkpoint — return untrained model with defaults
        model = RCNNSeparator(n_fft=512, n_sources=2, lstm_hidden=256, lstm_layers=2, dropout=0.0)
        model.eval()
        meta = {"n_fft": 512, "hop_length": 128}
        _loaded_models["rcnn"] = model
        _loaded_models["rcnn_meta"] = meta
        return model, meta

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

    meta = {"n_fft": n_fft, "hop_length": hop_length}
    _loaded_models["rcnn"] = model
    _loaded_models["rcnn_meta"] = meta
    return model, meta


def _load_audio(path: str, target_sr: int = 8000, max_duration: float = 30.0):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.squeeze(0)  # (T,)
    max_samples = int(target_sr * max_duration)
    if wav.shape[0] > max_samples:
        wav = wav[:max_samples]
    return wav


@router.post("/separate")
async def separate(
    audio: UploadFile = File(...),
    num_speakers: int = Form(2),
):
    device = "cpu"
    model, meta = _get_model(device)
    stft_helper = STFTHelper(n_fft=meta["n_fft"], hop_length=meta["hop_length"])

    suffix = Path(audio.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        waveform = _load_audio(tmp_path, target_sr=8000, max_duration=30.0)
        original_length = waveform.shape[0]
        segment_length = 32000

        if original_length < segment_length:
            waveform_padded = torch.nn.functional.pad(waveform, (0, segment_length - original_length))
        else:
            waveform_padded = waveform[:segment_length]

        waveform_padded = waveform_padded.to(device)
        mag, phase = stft_helper.stft(waveform_padded)
        mag_input = mag.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)

        with torch.no_grad():
            masks = model(mag_input)  # (1, n_sources, F, T)

        separated = []
        for s in range(masks.shape[1]):
            est_mag = masks[0, s] * mag
            est_wav = stft_helper.istft(est_mag, phase, length=min(original_length, segment_length))
            audio_np = normalize_waveform(est_wav.cpu()).numpy()
            separated.append(audio_np)

        output_files = []
        for i, audio_np in enumerate(separated):
            out_path = tmp_path + f"_speaker_{i+1}.wav"
            sf.write(out_path, audio_np, 8000, subtype="PCM_16")
            output_files.append(out_path)

        return {
            "model": "RCNN",
            "num_speakers": num_speakers,
            "files": output_files,
            "sample_rate": 8000,
        }
    finally:
        os.unlink(tmp_path)
