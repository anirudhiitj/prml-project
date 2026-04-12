# Cocktail Party Speech Separation

A web-based demo comparing **4 different approaches** to the cocktail party (audio source separation) problem.

| Model           | Approach                             | Author  |
| --------------- | ------------------------------------ | ------- |
| DPRNN-TasNet    | Dual-Path RNN                        | Anirudh |
| Conv-TasNet     | Convolutional TasNet                 | Mayank  |
| Conv-TasNet C++ | C++ with libtorch                    | Gokul   |
| RCNN            | Conv Encoder + BiLSTM + Conv Decoder | Mohit   |

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/<your-org>/prml-project.git
cd prml-project
```

### 2. Install dependencies

> **Python 3.10+** required. No GPU needed — CPU inference works fine.

```bash
pip install -r requirements.txt
```

**Note on PyTorch:** If you already have PyTorch installed, the above will use it. For a fresh CPU-only install:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **Windows users — if you get a build error for `pesq` or `pyroomacoustics`:**
>
> These packages have C extensions and require Microsoft C++ Build Tools to compile from source.
> They are only used by training/evaluation scripts — **the web app does not need them**.
> Either skip them or install pre-built wheels:
>
> ```bash
> pip install pesq --only-binary :all:
> pip install pyroomacoustics --only-binary :all:
> ```
>
> If no pre-built wheel exists for your Python version, just skip those two lines — `uvicorn app:app` will work without them.

> **If you want to use Mayank's Conv-TasNet (Asteroid) model (Linux or Windows):**
>
> Asteroid's default install pulls in C++ dependencies that often fail to build. Instead, run:
>
> ```bash
> pip install asteroid --no-deps
> pip install asteroid-filterbanks julius pytorch-lightning "torch-optimizer<0.2.0" "torchmetrics<=0.11.4"
> ```
>
> This works on both Windows and Linux. If you see errors about `pb_bss_eval` or `torch-stoi`, IGNORE them — they are not needed for inference.

If you only want to run the web app and not train/evaluate, you can skip Asteroid entirely (the Mayank model will just be unavailable in the UI).

### 3. Run the server

```bash
uvicorn app:app --reload --port 8000
```

### 4. Open the frontend

Go to **http://localhost:8000** in your browser.

1. Upload a mixed audio file (WAV, MP3, FLAC, OGG)
2. Select a model from the dropdown
3. Choose the number of speakers
4. Click **Separate**
5. Listen to / download the separated audio

## How It Works

- `app.py` — FastAPI backend that routes requests to each model's API
- `frontend/index.html` — Single-page UI
- `anirudh/`, `mayank/`, `mohit/`, `gokul/` — Each folder contains one approach with its own model, inference code, and API endpoint

**Checkpoint auto-merge:** Large model checkpoints are split into <100 MB parts (for GitHub's file size limit). On first inference, parts are automatically merged back — no manual step needed.

## Notes

- Models whose dependencies are not installed are **skipped gracefully** at startup (the server still runs, those models just show as unavailable).
- All inference runs on **CPU** — no GPU required.
- Audio is processed at **8 kHz** (mohit, mayank, gokul) or **16 kHz** (anirudh).

### Gokul's Conv-TasNet C++

No C++ compilation is needed. The model's checkpoint (`gokul/checkpoints/best_tasnet.pt`) was saved as a TorchScript archive from the C++ training code and is loaded directly by a matching pure-Python Conv-TasNet implementation (`gokul/model.py`).

If you are running on a Linux server where the C++ binary is compiled, you can enable the native binary instead:

```bash
USE_CPP_BINARY=1 uvicorn app:app --port 8000
```
