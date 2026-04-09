// ─────────────────────────────────────────────────────────────────────────────
// stft.cpp — Manual STFT/iSTFT using torch::fft::rfft
//
// We implement STFT manually because LibTorch's torch::stft dispatch symbol
// is not exported in prebuilt shared libraries (CompositeImplicitAutograd).
// The manual implementation is mathematically equivalent and also more
// educational for showing in a PRML report.
// ─────────────────────────────────────────────────────────────────────────────
#include "stft.h"
#include <cmath>

namespace stft_utils {

torch::Tensor hann_window(int64_t length) {
    return torch::hann_window(length, torch::kFloat32);
}

torch::Tensor stft(torch::Tensor waveform, const STFTConfig& cfg) {
    // Ensure 2D: [B, T]
    if (waveform.dim() == 1) waveform = waveform.unsqueeze(0);
    if (waveform.dim() == 3) waveform = waveform.squeeze(1);

    int64_t B = waveform.size(0);
    int64_t T = waveform.size(1);
    auto dev  = waveform.device();

    // Reflect-pad for centering (same as center=True)
    int64_t pad = cfg.n_fft / 2;
    waveform = torch::nn::functional::pad(waveform,
        torch::nn::functional::PadFuncOptions({pad, pad}).mode(torch::kReflect));

    int64_t T_padded = waveform.size(1);
    int64_t num_frames = (T_padded - cfg.win_len) / cfg.hop_len + 1;

    // Create Hann window
    auto window = hann_window(cfg.win_len).to(dev);

    // Extract overlapping frames using unfold: [B, num_frames, win_len]
    auto frames = waveform.unfold(1, cfg.win_len, cfg.hop_len);  // [B, num_frames, win_len]

    // Apply window
    frames = frames * window.unsqueeze(0).unsqueeze(0);  // broadcast [1, 1, win_len]

    // Zero-pad to n_fft if needed (win_len < n_fft)
    if (cfg.win_len < cfg.n_fft) {
        frames = torch::nn::functional::pad(frames,
            torch::nn::functional::PadFuncOptions({0, cfg.n_fft - cfg.win_len}));
    }

    // Compute real FFT → complex [B, num_frames, n_fft/2+1]
    auto spec = torch::fft::rfft(frames, cfg.n_fft);

    // Transpose to [B, F, T_frames]
    spec = spec.permute({0, 2, 1});

    return spec;  // Complex tensor [B, F, T_frames]
}

torch::Tensor istft(torch::Tensor spec, const STFTConfig& cfg, int64_t length) {
    // spec: Complex [B, F, T_frames]
    int64_t B = spec.size(0);
    int64_t F = spec.size(1);
    int64_t num_frames = spec.size(2);
    auto dev = spec.device();

    // Transpose to [B, T_frames, F]
    auto spec_t = spec.permute({0, 2, 1});

    // Inverse real FFT → [B, T_frames, n_fft]
    auto frames = torch::fft::irfft(spec_t, cfg.n_fft);

    // Trim to win_len (if n_fft > win_len)
    if (cfg.n_fft > cfg.win_len)
        frames = frames.index({torch::indexing::Slice(),
                               torch::indexing::Slice(),
                               torch::indexing::Slice(0, cfg.win_len)});

    // Create window for overlap-add normalization
    auto window = hann_window(cfg.win_len).to(dev);

    // Apply window
    auto windowed = frames * window.unsqueeze(0).unsqueeze(0);

    // Overlap-add synthesis
    int64_t pad = cfg.n_fft / 2;
    int64_t out_len = (num_frames - 1) * cfg.hop_len + cfg.win_len;
    auto output = torch::zeros({B, out_len}, spec.options().dtype(torch::kFloat32));
    auto win_sum = torch::zeros({out_len}, spec.options().dtype(torch::kFloat32));

    auto win_sq = window * window;

    for (int64_t t = 0; t < num_frames; ++t) {
        int64_t start = t * cfg.hop_len;
        auto sl = torch::indexing::Slice(start, start + cfg.win_len);
        output.index_put_({torch::indexing::Slice(), sl},
            output.index({torch::indexing::Slice(), sl}) + windowed.select(1, t));
        win_sum.index_put_({sl}, win_sum.index({sl}) + win_sq);
    }

    // Normalize by window sum (avoid division by zero)
    win_sum = torch::clamp(win_sum, 1e-8);
    output = output / win_sum.unsqueeze(0);

    // Remove centering padding
    output = output.index({torch::indexing::Slice(),
                           torch::indexing::Slice(pad, out_len - pad)});

    // Trim to desired length
    if (length > 0 && output.size(1) > length)
        output = output.index({torch::indexing::Slice(),
                               torch::indexing::Slice(0, length)});

    return output;  // [B, T]
}

torch::Tensor magnitude(torch::Tensor spec) {
    return torch::abs(spec);
}

torch::Tensor phase(torch::Tensor spec) {
    return torch::angle(spec);
}

torch::Tensor polar_to_complex(torch::Tensor mag, torch::Tensor phi) {
    auto real = mag * torch::cos(phi);
    auto imag = mag * torch::sin(phi);
    return torch::complex(real, imag);
}

}  // namespace stft_utils
