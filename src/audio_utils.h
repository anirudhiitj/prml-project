#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// audio_utils.h — WAV I/O using libsndfile
// ─────────────────────────────────────────────────────────────────────────────
#include <torch/torch.h>
#include <string>

namespace audio {

/// Load a WAV file into a float tensor.
/// Multi-channel files are averaged to mono.
/// @param path       Path to .wav file
/// @param target_sr  Expected sample rate (throws if mismatch)
/// @return           [1, T] float tensor
torch::Tensor load_wav(const std::string& path, int target_sr = 8000);

/// Save a float tensor as a mono WAV file.
/// @param path        Output path
/// @param waveform    [1, T] or [T] float tensor
/// @param sample_rate Sample rate in Hz
void save_wav(const std::string& path, torch::Tensor waveform, int sample_rate = 8000);

/// Pad or truncate waveform to exactly `length` samples.
torch::Tensor fix_length(torch::Tensor waveform, int64_t length);

}  // namespace audio
