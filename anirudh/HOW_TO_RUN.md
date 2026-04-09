# How to Run Inference with Best2 and Best3 Checkpoints

## Step 1: Install Requirements

From the `anirudh/cocktail_separation/` directory, run:

```bash
pip install -r requirements.txt
```

## Step 2: Run Inference

### For 2-Speaker Separation (Using best2.pt)

```bash
cd cocktail_separation
python -c "
import sys
sys.path.insert(0, '.')
from inference import *
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

device = torch.device('cpu')
mixture = load_audio('3sp1.mp3', max_duration=30.0)
model, cfg = load_model('checkpoints/best2.pt', 2, device=str(device))
estimates = separate_speakers(model, mixture, device=str(device))

amplification_factor = 4.0
output_path = Path('./my_results_2spk')
output_path.mkdir(parents=True, exist_ok=True)

for i, estimate in enumerate(estimates):
    audio_np = estimate.numpy()
    audio_np = audio_np * amplification_factor
    max_val = np.abs(audio_np).max()
    if max_val > 1.0:
        audio_np = audio_np / max_val
    sf.write(str(output_path / f'speaker_{i+1}.wav'), audio_np, 16000, subtype='PCM_16')
    print(f'✅ Speaker {i+1}: speaker_{i+1}.wav (amplified {amplification_factor}x)')
"
```

### For 3-Speaker Separation (Using best3.pt)

```bash
cd cocktail_separation
python -c "
import sys
sys.path.insert(0, '.')
from inference import *
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

device = torch.device('cpu')
mixture = load_audio('3sp1.mp3', max_duration=30.0)
model, cfg = load_model('checkpoints/best3.pt', 3, device=str(device))
estimates = separate_speakers(model, mixture, device=str(device))

amplification_factor = 4.0
output_path = Path('./my_results_3spk')
output_path.mkdir(parents=True, exist_ok=True)

for i, estimate in enumerate(estimates):
    audio_np = estimate.numpy()
    audio_np = audio_np * amplification_factor
    max_val = np.abs(audio_np).max()
    if max_val > 1.0:
        audio_np = audio_np / max_val
    sf.write(str(output_path / f'speaker_{i+1}.wav'), audio_np, 16000, subtype='PCM_16')
    print(f'✅ Speaker {i+1}: speaker_{i+1}.wav (amplified {amplification_factor}x)')
"
```

## Key Changes

- **best2.pt**: Used for 2-speaker separation (num_spk=2)
  - Output saved to: `./my_results_2spk/`
  - Extracts 2 speaker sources from mixed audio

- **best3.pt**: Used for 3-speaker separation (num_spk=3)
  - Output saved to: `./my_results_3spk/`
  - Extracts 3 speaker sources from mixed audio

## Notes

- Ensure `3sp1.mp3` exists in the `cocktail_separation/` directory
- Results are amplified by 4x and normalized to prevent clipping
- Output audio is saved in 16kHz PCM_16 format
- Use CPU mode (`device='cpu'`) for compatibility
