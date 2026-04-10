"""FastAPI endpoint for DPRNN-TasNet (Anirudh's model)."""

import sys
import os
import tempfile
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

# Ensure src/ is importable
_dir = os.path.dirname(__file__)
if _dir not in sys.path:
    sys.path.insert(0, _dir)
from src.model import DPRNNTasNet

router = APIRouter(prefix="/anirudh", tags=["DPRNN-TasNet"])

# Checkpoint paths (relative to this file)
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS = {
    "2spk": BASE_DIR / "checkpoints" / "best2.pt",
    "3spk": BASE_DIR / "checkpoints" / "best3.pt",
    "5spk": BASE_DIR / "checkpoints" / "best.pt",
}

_loaded_models = {}


def _auto_merge_checkpoint(checkpoint_path: Path) -> None:
    if checkpoint_path.exists():
        return
    parts = sorted(checkpoint_path.parent.glob(checkpoint_path.name + ".part*"),
                   key=lambda p: int(p.suffix.lstrip(".part")))
    if not parts:
        return
    print(f"🔧 Auto-merging {len(parts)} parts → {checkpoint_path.name}")
    with open(checkpoint_path, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
    print(f"✅ Merged: {checkpoint_path.name} ({checkpoint_path.stat().st_size / 1024**2:.1f} MB)")


def _get_model(num_speakers: int, device: str = "cpu"):
    key_map = {2: "2spk", 3: "3spk", 5: "5spk"}
    key = key_map.get(num_speakers)
    if key is None or not CHECKPOINTS.get(key, Path("_")).parent.exists():
        return None, f"No checkpoint for {num_speakers} speakers"

    ckpt_path = CHECKPOINTS.get(key, Path("_"))
    _auto_merge_checkpoint(ckpt_path)

    if not ckpt_path.exists():
        return None, f"No checkpoint for {num_speakers} speakers"

    if key not in _loaded_models:
        ckpt = torch.load(str(CHECKPOINTS[key]), map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = DPRNNTasNet(
            num_speakers=int(cfg["model"]["num_speakers"]),
            encoder_dim=int(cfg["model"]["encoder_dim"]),
            encoder_kernel=int(cfg["model"]["encoder_kernel"]),
            encoder_stride=int(cfg["model"]["encoder_stride"]),
            bottleneck_dim=int(cfg["model"]["bottleneck_dim"]),
            chunk_size=int(cfg["model"]["chunk_size"]),
            num_dprnn_blocks=int(cfg["model"]["num_dprnn_blocks"]),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _loaded_models[key] = model

    return _loaded_models[key], None


def _load_audio(path: str, sample_rate: int = 16000, max_duration: float = 30.0):
    import librosa
    waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
    max_samples = int(sample_rate * max_duration)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    return torch.from_numpy(waveform).unsqueeze(0).float()


@router.post("/separate")
async def separate(
    audio: UploadFile = File(...),
    num_speakers: int = Form(2),
):
    device = "cpu"
    model, err = _get_model(num_speakers, device)
    if err:
        return JSONResponse(status_code=400, content={"error": err})

    # Save uploaded file to temp
    suffix = Path(audio.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        mixture = _load_audio(tmp_path, sample_rate=16000, max_duration=30.0)

        with torch.no_grad():
            estimates = model(mixture.to(device))  # (1, C, T)
            estimates = estimates.squeeze(0).cpu()  # (C, T)

        # Amplify and save to temp files
        output_files = []
        for i in range(estimates.shape[0]):
            audio_np = estimates[i].numpy() * 4.0
            peak = np.abs(audio_np).max()
            if peak > 1.0:
                audio_np = audio_np / peak
            out_path = tmp_path + f"_speaker_{i+1}.wav"
            sf.write(out_path, audio_np, 16000, subtype="PCM_16")
            output_files.append(out_path)

        return {
            "model": "DPRNN-TasNet",
            "num_speakers": num_speakers,
            "files": output_files,
            "sample_rate": 16000,
        }
    finally:
        os.unlink(tmp_path)
