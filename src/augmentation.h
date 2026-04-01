#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// augmentation.h — Data augmentation for speech separation
// ─────────────────────────────────────────────────────────────────────────────
#include <torch/torch.h>

namespace augment {

struct AugConfig {
    bool   enable_gain   = true;
    bool   enable_noise  = true;
    bool   enable_shift  = true;
    bool   enable_flip   = true;
    float  gain_db       = 6.0f;
    float  noise_snr_lo  = 20.0f;
    float  noise_snr_hi  = 40.0f;
    float  shift_frac    = 0.1f;
    float  prob          = 0.5f;
};

/// Apply augmentations to mixture + sources consistently.
/// @param mixture  [1, T]
/// @param sources  [C, T]
/// @param cfg      augmentation config
/// @return         (augmented_mix, augmented_sources)
std::pair<torch::Tensor, torch::Tensor>
apply(torch::Tensor mixture, torch::Tensor sources,
      const AugConfig& cfg = {});

}  // namespace augment
