// ─────────────────────────────────────────────────────────────────────────────
// metrics.cpp — Evaluation metrics implementation
// ─────────────────────────────────────────────────────────────────────────────
#include "metrics.h"
#include "losses.h"
#include <cmath>
#include <algorithm>

namespace metrics {

float si_snr_improvement(torch::Tensor sep, torch::Tensor tgt,
                          torch::Tensor mix) {
    sep = sep.contiguous().to(torch::kCPU);
    tgt = tgt.contiguous().to(torch::kCPU);
    mix = mix.contiguous().to(torch::kCPU);

    float snr_out = losses::si_snr(sep, tgt).item<float>();
    float snr_in  = losses::si_snr(mix, tgt).item<float>();
    return snr_out - snr_in;
}

float sdr(torch::Tensor est, torch::Tensor tgt) {
    est = est.contiguous().to(torch::kCPU, torch::kFloat32).squeeze();
    tgt = tgt.contiguous().to(torch::kCPU, torch::kFloat32).squeeze();

    // SDR = 10 * log10( ||s||^2 / ||s - s'||^2 )
    auto noise = tgt - est;
    float sig_pow   = (tgt * tgt).sum().item<float>();
    float noise_pow = (noise * noise).sum().item<float>();
    if (noise_pow < 1e-10f) return 100.0f;  // perfect
    return 10.0f * std::log10(sig_pow / (noise_pow + 1e-10f));
}

float sdr_improvement(torch::Tensor sep, torch::Tensor tgt,
                       torch::Tensor mix) {
    return sdr(sep, tgt) - sdr(mix, tgt);
}

float stoi_estimate(torch::Tensor est, torch::Tensor tgt, int sr) {
    est = est.contiguous().to(torch::kCPU, torch::kFloat32).squeeze();
    tgt = tgt.contiguous().to(torch::kCPU, torch::kFloat32).squeeze();

    // Simplified STOI: frame-level cross-correlation
    // Frame size: 30ms window (standard STOI uses 384 samples at 10kHz)
    int frame_len = sr * 30 / 1000;  // 30ms
    if (frame_len <= 0) frame_len = 240;

    int64_t T = std::min(est.size(0), tgt.size(0));
    int64_t num_frames = T / frame_len;
    if (num_frames <= 0) return 0.0f;

    auto est_frames = est.index({torch::indexing::Slice(0, num_frames * frame_len)})
                         .view({num_frames, frame_len});
    auto tgt_frames = tgt.index({torch::indexing::Slice(0, num_frames * frame_len)})
                         .view({num_frames, frame_len});

    // Normalized cross-correlation per frame
    auto est_norm = est_frames - est_frames.mean(1, true);
    auto tgt_norm = tgt_frames - tgt_frames.mean(1, true);

    auto est_std = est_norm.pow(2).sum(1).sqrt() + 1e-8;
    auto tgt_std = tgt_norm.pow(2).sum(1).sqrt() + 1e-8;

    auto corr = (est_norm * tgt_norm).sum(1) / (est_std * tgt_std);

    // Clamp correlation to [-1, 1] and average
    corr = torch::clamp(corr, -1.0, 1.0);
    float mean_corr = corr.mean().item<float>();

    // Map from [-1, 1] correlation to [0, 1] STOI-like score
    return std::clamp((mean_corr + 1.0f) / 2.0f, 0.0f, 1.0f);
}

MetricResult evaluate(torch::Tensor sep, torch::Tensor tgt,
                       torch::Tensor mix, int sr) {
    return {
        si_snr_improvement(sep, tgt, mix),
        sdr_improvement(sep, tgt, mix),
        stoi_estimate(sep, tgt, sr)
    };
}

}  // namespace metrics
