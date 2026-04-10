"""
Main FastAPI application — Cocktail Party Speech Separation

Routes audio to the selected model's endpoint, returns separated audio files
and serves the frontend.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup temp files tracked during session
    for f in _temp_files:
        try:
            os.unlink(f)
        except OSError:
            pass

app = FastAPI(
    title="Cocktail Party Speech Separation",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_temp_files: list[str] = []

# ───── Import routers via importlib to avoid sys.path conflicts ─────
import importlib.util

def _import_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_apis = {}

_api_configs = {
    "anirudh": ("anirudh_api", str(ROOT / "anirudh" / "cocktail_separation" / "api.py")),
    "mayank": ("mayank_api", str(ROOT / "mayank" / "api.py")),
    "mohit": ("mohit_api", str(ROOT / "mohit" / "api.py")),
    "gokul": ("gokul_api", str(ROOT / "gokul" / "api.py")),
}

for _key, (_mod_name, _mod_path) in _api_configs.items():
    try:
        _mod = _import_module_from_path(_mod_name, _mod_path)
        _apis[_key] = _mod
        app.include_router(_mod.router)
        print(f"  ✔ Loaded {_key} API")
    except Exception as exc:
        print(f"  ⚠ Skipped {_key} API: {exc}")

anirudh_api = _apis.get("anirudh")
mayank_api = _apis.get("mayank")
mohit_api = _apis.get("mohit")
gokul_api = _apis.get("gokul")


# ───── Unified separation endpoint ─────
@app.post("/api/separate")
async def separate(
    audio: UploadFile = File(...),
    model: str = Form("dprnn"),
    num_speakers: int = Form(2),
):
    """Unified endpoint that delegates to the selected model."""

    # Map model names to API modules
    model_api_map = {
        "dprnn": anirudh_api,
        "convtasnet": mayank_api,
        "rcnn": mohit_api,
        "convtasnet_cpp": gokul_api,
    }

    api_mod = model_api_map.get(model)
    if api_mod is None:
        available = [k for k, v in model_api_map.items() if v is not None]
        return JSONResponse(
            status_code=400,
            content={"error": f"Model '{model}' is not available. Available: {available}"},
        )

    # Forward to the specific router
    content = await audio.read()
    await audio.seek(0)

    # Direct function dispatch
    audio.file.seek(0)
    result = await api_mod.separate(audio=audio, num_speakers=num_speakers)

    if isinstance(result, JSONResponse):
        return result

    # Track temp files for cleanup
    for f in result.get("files", []):
        _temp_files.append(f)

    return result


@app.get("/api/download")
async def download_file(path: str):
    """Download a separated audio or image file by its temp path."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    media_map = {".wav": "audio/wav", ".png": "image/png", ".jpg": "image/jpeg"}
    media_type = media_map.get(p.suffix.lower(), "application/octet-stream")
    return FileResponse(str(p), media_type=media_type, filename=p.name)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = ROOT / "frontend" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok"}
