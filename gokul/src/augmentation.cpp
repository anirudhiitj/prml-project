// ─────────────────────────────────────────────────────────────────────────────
// augmentation.cpp — Data augmentation implementation
// ─────────────────────────────────────────────────────────────────────────────
#include "augmentation.h"
#include <random>
#include <cmath>

namespace augment {

static thread_local std::mt19937 rng{std::random_device{}()};
static bool coin(float p) { return std::uniform_real_distribution<float>(0,1)(rng) < p; }
static float unif(float a, float b) { return std::uniform_real_distribution<float>(a,b)(rng); }

std::pair<torch::Tensor, torch::Tensor>
apply(torch::Tensor mix, torch::Tensor src, const AugConfig& cfg) {
    // 1. Random gain (applied to both so targets stay consistent)
    if (cfg.enable_gain && coin(cfg.prob)) {
        float g = std::pow(10.0f, unif(-cfg.gain_db, cfg.gain_db) / 20.0f);
        mix = mix * g;
        src = src * g;
    }

    // 2. Additive Gaussian noise (mixture only — sources remain clean)
    if (cfg.enable_noise && coin(cfg.prob)) {
        float snr_db = unif(cfg.noise_snr_lo, cfg.noise_snr_hi);
        float sig_pow = mix.pow(2).mean().item<float>();
        float noise_std = std::sqrt(sig_pow / std::pow(10.0f, snr_db / 10.0f));
        mix = mix + torch::randn_like(mix) * noise_std;
    }

    // 3. Circular time shift (both)
    if (cfg.enable_shift && coin(cfg.prob)) {
        int64_t T = mix.size(-1);
        int64_t max_s = static_cast<int64_t>(cfg.shift_frac * T);
        if (max_s > 0) {
            int64_t s = std::uniform_int_distribution<int64_t>(-max_s, max_s)(rng);
            mix = torch::roll(mix, s, -1);
            src = torch::roll(src, s, -1);
        }
    }

    // 4. Polarity flip (both)
    if (cfg.enable_flip && coin(cfg.prob)) {
        mix = -mix;
        src = -src;
    }

    return {mix, src};
}

}  // namespace augment
