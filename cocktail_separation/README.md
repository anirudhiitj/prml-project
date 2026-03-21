# Cocktail Separation (DPRNN-TasNet)

PyTorch implementation of a DPRNN-TasNet system for multi-speaker blind source separation.

## Project Structure

- configs: experiment configs for 2/3/4/5 speaker curriculum
- src: model, separator, losses, and dataset code
- scripts: mixture generation and RIR preparation
- train.py: DDP-aware training entrypoint
- evaluate.py: SI-SNRi/SDRi/PESQ evaluation
- separate.py: inference on a single input waveform

## Install

```bash
pip install -r requirements.txt
```

## Generate Mixtures

```bash
python scripts/generate_mixtures.py \
  --source_root data/raw/speech_sources \
  --output_root data/mixtures/5spk/train \
  --num_mixtures 100000 \
  --num_speakers 5
```

Repeat for val/test and each curriculum stage (2spk/3spk/4spk/5spk).

## Train (Single Node, 8 GPUs)

```bash
torchrun --nproc_per_node=8 train.py --config configs/5spk.yaml
```

## Evaluate

```bash
python evaluate.py --config configs/5spk.yaml --checkpoint checkpoints/best.pt
```

## Inference

```bash
python separate.py \
  --config configs/5spk.yaml \
  --checkpoint checkpoints/best.pt \
  --input path/to/mixture.wav \
  --output_dir outputs/
```

## Notes

- PIT uses Hungarian assignment at utterance level.
- Masking uses softmax across speakers.
- Curriculum training is expected: 2 -> 3 -> 4 -> 5 speakers.
