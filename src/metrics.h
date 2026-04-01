#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// metrics.h — Evaluation metrics for speech separation
// ─────────────────────────────────────────────────────────────────────────────
#include <torch/torch.h>

namespace metrics {

/// SI-SNR improvement: SI-SNR(separated, target) - SI-SNR(mixture, target)
/// All inputs [T] or [B, T]
float si_snr_improvement(torch::Tensor separated, torch::Tensor target,
                          torch::Tensor mixture);

/// Signal-to-Distortion Ratio (BSSEval-style simplified)
/// estimate, target: [T]
float sdr(torch::Tensor estimate, torch::Tensor target);

/// SDR improvement
float sdr_improvement(torch::Tensor separated, torch::Tensor target,
                       torch::Tensor mixture);

/// Simplified STOI estimation (short-time objective intelligibility)
/// Based on short-time cross-correlation between clean and processed signals.
/// estimate, target: [T], sr: sample rate
float stoi_estimate(torch::Tensor estimate, torch::Tensor target, int sr = 8000);

/// Compute all metrics for a separation result and return as struct.
struct MetricResult {
    float si_snri;   // SI-SNR improvement (dB)
    float sdri;      // SDR improvement (dB)
    float stoi;      // STOI estimate (0-1)
};

MetricResult evaluate(torch::Tensor separated, torch::Tensor target,
                       torch::Tensor mixture, int sr = 8000);

}  // namespace metrics
