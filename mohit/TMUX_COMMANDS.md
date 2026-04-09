# Tmux Commands for RCNN Training

## Your Training Session

| Item | Value |
|------|-------|
| **Session name** | `rcnn_train` |
| **GPU** | GPU 7 (NVIDIA H200) |
| **Conda env** | `rl4vlm_clean` |
| **Log file** | `training.log` (in project dir) |

---

## Essential Tmux Commands

### Viewing the Training

```bash
# Attach to the training session (see live output)
tmux attach -t rcnn_train

# Detach from session (training continues in background)
# Press: Ctrl+B, then D

# View session output without attaching
tmux capture-pane -t rcnn_train -p | tail -30

# View training log file directly
tail -f /mnt/raid/rl_gaming/LLM4DyG-forked2/cocktail_party_rcnn/training.log
```

### Session Management

```bash
# List all tmux sessions
tmux ls

# Kill the training session (STOPS training)
tmux kill-session -t rcnn_train

# Create a new session
tmux new-session -s <session_name>
```

### Inside a Tmux Session (Key Shortcuts)

| Shortcut | Action |
|----------|--------|
| `Ctrl+B`, then `D` | **Detach** — exit tmux, training keeps running |
| `Ctrl+B`, then `[` | **Scroll mode** — use arrow keys/PgUp to scroll, press `q` to exit |
| `Ctrl+B`, then `c` | Create a new window |
| `Ctrl+B`, then `n` | Next window |
| `Ctrl+B`, then `p` | Previous window |
| `Ctrl+C` | **Kill the running process** (stops training!) |

### Checking GPU Usage

```bash
# Check if GPU 7 is being used
nvidia-smi | grep -A4 "GPU 7"

# Watch GPU usage continuously
watch -n 2 nvidia-smi
```

### After Training Completes

```bash
# Run inference on an audio file
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl4vlm_clean
cd /mnt/raid/rl_gaming/LLM4DyG-forked2/cocktail_party_rcnn

CUDA_VISIBLE_DEVICES=7 python inference.py \
    --input path/to/mixed_audio.wav \
    --checkpoint ./checkpoints/best_model.pt \
    --output_dir ./separated_output
```

---

> **Tip**: You can safely close your laptop. The tmux session keeps running on the server. Just SSH back in and run `tmux attach -t rcnn_train` to check progress.
