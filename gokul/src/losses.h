#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// losses.h — SI-SNR loss with PIT + spectral loss for U-Net
// ─────────────────────────────────────────────────────────────────────────────
#include <torch/torch.h>

namespace losses {

/// SI-SNR in dB.  estimate, target: [T] or [B, T]
torch::Tensor si_snr(torch::Tensor estimate, torch::Tensor target);

/// PIT loss for 2-speaker separation (negative mean SI-SNR, best permutation).
/// estimates, targets: [B, C, T]
torch::Tensor pit_loss(torch::Tensor estimates, torch::Tensor targets);

/// L1 spectral loss on magnitude spectrograms (for U-Net baseline).
/// Computes PIT over spectrogram masks.
/// est_masks: [B, C, F, T], mix_mag: [B, 1, F, T], tgt_mags: [B, C, F, T]
torch::Tensor spectral_pit_loss(torch::Tensor est_masks,
                                 torch::Tensor mix_mag,
                                 torch::Tensor tgt_mags);

}  // namespace losses
