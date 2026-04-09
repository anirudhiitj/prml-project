#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// preprocessing.h — Audio preprocessing pipeline
//
// Steps: mono conversion → peak normalization → silence trimming (VAD)
//
// Design rationale:
//   - We normalize amplitude so the network sees consistent levels.
//   - We trim silence so training segments contain actual speech.
//   - We do NOT apply aggressive filtering or denoising, because that
//     destroys speech harmonics and formant structure that the separator needs.
//   - Any denoising should happen *after* separation if needed downstream.
// ─────────────────────────────────────────────────────────────────────────────
#include <torch/torch.h>

namespace preprocess {

struct PreprocessConfig {
    float  peak_amplitude   = 0.9f;    // target peak after normalization
    float  vad_energy_db    = -40.0f;  // silence threshold in dB below peak
    int    vad_frame_ms     = 20;      // VAD frame size in milliseconds
    int    vad_min_speech_ms = 200;    // minimum speech duration to keep
    int    sample_rate      = 8000;
};

/// Peak-normalize a waveform to target amplitude.
/// @param waveform [1, T] or [T]
/// @param target   target peak level (e.g. 0.9)
/// @return         normalized waveform, same shape
torch::Tensor normalize(torch::Tensor waveform, float target = 0.9f);

/// Remove leading/trailing silence based on frame energy.
/// Uses a simple energy-based voice activity detector.
/// @param waveform [1, T]
/// @param cfg      preprocessing config
/// @return         trimmed waveform [1, T']  where T' ≤ T
torch::Tensor vad_trim(torch::Tensor waveform, const PreprocessConfig& cfg = {});

/// Full preprocessing pipeline: normalize → trim silence.
/// @param waveform [1, T]
/// @param cfg      config
/// @return         preprocessed waveform [1, T']
torch::Tensor preprocess(torch::Tensor waveform, const PreprocessConfig& cfg = {});

}  // namespace preprocess
