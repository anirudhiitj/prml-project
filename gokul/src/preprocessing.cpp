// ─────────────────────────────────────────────────────────────────────────────
// preprocessing.cpp — Audio preprocessing implementation
// ─────────────────────────────────────────────────────────────────────────────
#include "preprocessing.h"
#include <cmath>
#include <algorithm>

namespace preprocess {

torch::Tensor normalize(torch::Tensor waveform, float target) {
    auto peak = waveform.abs().max();
    float peak_val = peak.item<float>();
    if (peak_val < 1e-8f) return waveform;  // silence — don't amplify noise
    return waveform * (target / peak_val);
}

torch::Tensor vad_trim(torch::Tensor waveform, const PreprocessConfig& cfg) {
    if (waveform.dim() == 1) waveform = waveform.unsqueeze(0);
    int64_t T = waveform.size(1);

    // Frame-level energy computation
    int frame_size = (cfg.sample_rate * cfg.vad_frame_ms) / 1000;
    if (frame_size <= 0) frame_size = 160;

    int64_t num_frames = T / frame_size;
    if (num_frames <= 0) return waveform;

    // Compute energy per frame in dB
    auto frames = waveform.squeeze(0)
                      .index({torch::indexing::Slice(0, num_frames * frame_size)})
                      .view({num_frames, frame_size});
    auto energy = (frames * frames).mean(1);  // [num_frames]
    auto energy_db = 10.0f * torch::log10(energy + 1e-10f);

    // Threshold: peak energy + vad_energy_db
    float peak_energy_db = energy_db.max().item<float>();
    float threshold = peak_energy_db + cfg.vad_energy_db;

    // Find first and last frame above threshold
    auto active = energy_db > threshold;
    auto active_acc = active.to(torch::kFloat32);

    // Find boundaries
    int64_t first_frame = 0;
    int64_t last_frame = num_frames - 1;

    auto active_ptr = active.data_ptr<bool>();
    for (int64_t i = 0; i < num_frames; ++i) {
        if (active_ptr[i]) { first_frame = i; break; }
    }
    for (int64_t i = num_frames - 1; i >= 0; --i) {
        if (active_ptr[i]) { last_frame = i; break; }
    }

    // Check minimum speech duration
    int min_frames = (cfg.sample_rate * cfg.vad_min_speech_ms) / (1000 * frame_size);
    if ((last_frame - first_frame + 1) < min_frames) {
        return waveform;  // too short, return unchanged
    }

    int64_t start_sample = first_frame * frame_size;
    int64_t end_sample   = std::min((last_frame + 1) * frame_size, T);

    return waveform.index({torch::indexing::Slice(),
                           torch::indexing::Slice(start_sample, end_sample)});
}

torch::Tensor preprocess(torch::Tensor waveform, const PreprocessConfig& cfg) {
    if (waveform.dim() == 1) waveform = waveform.unsqueeze(0);

    // Step 1: Peak normalization
    waveform = normalize(waveform, cfg.peak_amplitude);

    // Step 2: Silence trimming
    waveform = vad_trim(waveform, cfg);

    return waveform;
}

}  // namespace preprocess
