# DPRNN-TasNet Training Log
**Start Date**: 2026-03-22
**Status**: Initializing
**GPUs**: 5, 6 (143GB each)

## Phases Overview
| Phase | Speakers | Status | Start Time | End Time | Best Val SI-SNR | Notes |
|-------|----------|--------|-----------|---------|----------------|-------|
| 1 | 2-speaker | Not Started | - | - | - | 80k train, 4k val, 2k test |
| 2 | 3-speaker | Not Started | - | - | - | 80k train, 4k val, 2k test |
| 3 | 4-speaker | Not Started | - | - | - | 80k train, 4k val, 2k test |
| 4 | 5-speaker | Not Started | - | - | - | 100k train, 5k val, 3k test |

## Current Phase: 1 (2-speaker)
### Configuration
- **Model**: DPRNNTasNet (N=64, H=64, K=100, 6 blocks)
- **Batch Size**: 8 (per GPU, 16 total on GPU 5,6)
- **Learning Rate**: 1e-3
- **Optimizer**: Adam (β1=0.9, β2=0.999, weight_decay=1e-5)
- **Data**: 2-speaker mixtures (4 sec, 16kHz)
- **Augmentation**: RIR (0.7), Noise (0.5), Speed (0.3)

### Training Progress
```
Epoch: [Starting]
Batch: [...]
ETA: [Calculating...]
Val SI-SNR: [...]
Learning Rate: [...]
```

## Data Status
- [ ] LibriSpeech train-clean-360 downloaded
- [ ] LibriSpeech train-clean-100 downloaded
- [ ] 2-speaker mixtures generated (80k train)
- [ ] 3-speaker mixtures generated (80k train)
- [ ] 4-speaker mixtures generated (80k train)
- [ ] 5-speaker mixtures generated (100k train)

## Important Notes
- Using curriculum learning: 2→3→4→5 speakers
- Utterance-level PIT with Hungarian algorithm
- SI-SNR loss + 0.1 × SNR loss for better amplitude tracking
- Gradient clipping at norm=5.0
- Checkpoints saved: latest.pt, best.pt, epoch_N.pt (keep last 3)
- wandb logging enabled for real-time monitoring
