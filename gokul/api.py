"""FastAPI endpoint for Conv-TasNet C++ (Gokul's model).

Gokul's model is C++ based — we shell out to the compiled binary for inference.
Ensure the binary is compiled first via CMake.
"""

import os
import tempfile
import subprocess
import shutil
import soundfile as sf
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/gokul", tags=["Conv-TasNet-CPP"])

BASE_DIR = Path(__file__).resolve().parent
# The compiled inference binary (adjust name if different on your system)
INFERENCE_BIN = BASE_DIR / "build" / "inference"
if os.name == "nt":
    INFERENCE_BIN = BASE_DIR / "build" / "Release" / "inference.exe"
    if not INFERENCE_BIN.exists():
        INFERENCE_BIN = BASE_DIR / "build" / "inference.exe"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"


def _find_checkpoint(model_type: str = "tasnet"):
    if not CHECKPOINT_DIR.exists():
        return None
    for f in sorted(CHECKPOINT_DIR.glob(f"*{model_type}*.pt")):
        return f
    for f in sorted(CHECKPOINT_DIR.glob("*.pt")):
        return f
    return None


@router.post("/separate")
async def separate(
    audio: UploadFile = File(...),
    num_speakers: int = Form(2),
    model_type: str = Form("tasnet"),
):
    if not INFERENCE_BIN.exists():
        return JSONResponse(
            status_code=503,
            content={
                "error": f"C++ inference binary not found at {INFERENCE_BIN}. "
                         "Please compile with CMake first."
            },
        )

    ckpt = _find_checkpoint(model_type)
    if ckpt is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No checkpoint found in gokul/checkpoints/"},
        )

    suffix = Path(audio.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    output_dir = tempfile.mkdtemp()

    try:
        cmd = [
            str(INFERENCE_BIN),
            "--model", model_type,
            "--checkpoint", str(ckpt),
            "--input", tmp_path,
            "--output_dir", output_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={"error": f"Inference failed: {result.stderr}"},
            )

        output_files = sorted(Path(output_dir).glob("*.wav"))
        # Move to more stable temp paths so they persist for download
        stable_files = []
        for f in output_files:
            stable = tmp_path + f"_{f.stem}.wav"
            shutil.copy2(str(f), stable)
            stable_files.append(stable)

        return {
            "model": f"Conv-TasNet-CPP ({model_type})",
            "num_speakers": num_speakers,
            "files": stable_files,
            "sample_rate": 8000,
        }
    finally:
        os.unlink(tmp_path)
        shutil.rmtree(output_dir, ignore_errors=True)
