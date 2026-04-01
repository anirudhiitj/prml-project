#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// stft.h — Short-Time Fourier Transform utilities
//
// Design:
//   Window = 256 samples (32 ms @ 8 kHz) → captures ~3 pitch periods
//   Hop    = 64 samples  (8 ms)          → 75% overlap for smooth OLA
//   FFT    = 256                          → equal to window at 8 kHz
//   Window function: Hann → best sidelobe suppression for speech
//
// Why STFT over plain FFT?
//   A speech signal is non-stationary: phonemes, pitch, and energy change
//   rapidly. Plain FFT gives one global spectrum — useless for separating
//   speakers who alternate in time-frequency. STFT gives a 2D (time × freq)
//   representation where each speaker occupies distinct T-F regions.
//   The separator can then learn T-F masks.
// ─────────────────────────────────────────────────────────────────────────────
#include <torch/torch.h>

namespace stft_utils {

struct STFTConfig {
    int64_t n_fft     = 256;    // FFT size
    int64_t win_len   = 256;    // window length in samples
    int64_t hop_len   = 64;     // hop length in samples
    // Uses Hann window internally
};

/// Compute STFT of a waveform.
/// @param waveform  [B, T] or [1, T]
/// @param cfg       STFT parameters
/// @return          Complex tensor [B, F, T_frames] where F = n_fft/2 + 1
torch::Tensor stft(torch::Tensor waveform, const STFTConfig& cfg = {});

/// Inverse STFT: reconstruct waveform from complex spectrogram.
/// @param spec      Complex tensor [B, F, T_frames]
/// @param cfg       STFT parameters (must match forward STFT)
/// @param length    Optional: desired output length for trimming
/// @return          [B, T] waveform
torch::Tensor istft(torch::Tensor spec, const STFTConfig& cfg = {},
                    int64_t length = -1);

/// Extract magnitude spectrogram from complex STFT output.
/// @param spec  Complex tensor [B, F, T_frames]
/// @return      [B, F, T_frames] magnitude
torch::Tensor magnitude(torch::Tensor spec);

/// Extract phase from complex STFT output.
/// @param spec  Complex tensor [B, F, T_frames]
/// @return      [B, F, T_frames] phase in radians
torch::Tensor phase(torch::Tensor spec);

/// Reconstruct complex spectrogram from magnitude and phase.
/// @return Complex tensor [B, F, T_frames]
torch::Tensor polar_to_complex(torch::Tensor mag, torch::Tensor phi);

/// Create a Hann window of given length.
torch::Tensor hann_window(int64_t length);

}  // namespace stft_utils
