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

anirudh_api = _import_module_from_path("anirudh_api", str(ROOT / "anirudh" / "cocktail_separation" / "api.py"))
mayank_api = _import_module_from_path("mayank_api", str(ROOT / "mayank" / "api.py"))
mohit_api = _import_module_from_path("mohit_api", str(ROOT / "mohit" / "api.py"))
gokul_api = _import_module_from_path("gokul_api", str(ROOT / "gokul" / "api.py"))

anirudh_router = anirudh_api.router
mayank_router = mayank_api.router
mohit_router = mohit_api.router
gokul_router = gokul_api.router

app.include_router(anirudh_router)
app.include_router(mayank_router)
app.include_router(mohit_router)
app.include_router(gokul_router)


# ───── Unified separation endpoint ─────
@app.post("/api/separate")
async def separate(
    audio: UploadFile = File(...),
    model: str = Form("dprnn"),
    num_speakers: int = Form(2),
):
    """Unified endpoint that delegates to the selected model."""
    from fastapi.testclient import TestClient

    # Map model names to internal endpoints
    route_map = {
        "dprnn": "/anirudh/separate",
        "convtasnet": "/mayank/separate",
        "convtasnet_cpp": "/gokul/separate",
        "rcnn": "/mohit/separate",
    }

    target = route_map.get(model)
    if not target:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown model '{model}'. Options: {list(route_map.keys())}"},
        )

    # Forward to the specific router
    content = await audio.read()
    await audio.seek(0)

    # Direct function dispatch instead of internal HTTP call
    if model == "dprnn":
        fn = anirudh_api.separate
        audio.file.seek(0)
        result = await fn(audio=audio, num_speakers=num_speakers)
    elif model == "convtasnet":
        fn = mayank_api.separate
        audio.file.seek(0)
        result = await fn(audio=audio, num_speakers=num_speakers)
    elif model == "rcnn":
        fn = mohit_api.separate
        audio.file.seek(0)
        result = await fn(audio=audio, num_speakers=num_speakers)
    elif model == "convtasnet_cpp":
        fn = gokul_api.separate
        audio.file.seek(0)
        result = await fn(audio=audio, num_speakers=num_speakers)

    if isinstance(result, JSONResponse):
        return result

    # Track temp files for cleanup
    for f in result.get("files", []):
        _temp_files.append(f)

    return result


@app.get("/api/download")
async def download_file(path: str):
    """Download a separated audio file by its temp path."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(str(p), media_type="audio/wav", filename=p.name)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = ROOT / "frontend" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok"}
