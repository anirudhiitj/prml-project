#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// unet.h — Spectrogram U-Net baseline for speech separation
//
// Classical approach: STFT → learn magnitude masks → apply mask → iSTFT
//
// This serves as the STFT-based baseline to compare against Conv-TasNet.
// It makes the Fourier analysis explicit and interpretable.
//
// Architecture:
//   Input:  magnitude spectrogram [B, 1, F, T_frames]
//   Encoder: 4 downsampling blocks (Conv2d + BN + ReLU + MaxPool)
//   Bottleneck: Conv2d block
//   Decoder: 4 upsampling blocks (ConvTranspose2d + skip concat + Conv2d + BN + ReLU)
//   Output: C masks [B, C, F, T_frames] (sigmoid-bounded)
// ─────────────────────────────────────────────────────────────────────────────
#include <torch/torch.h>

namespace model {

struct UNetBlockImpl : torch::nn::Module {
    UNetBlockImpl(int64_t in_ch, int64_t out_ch);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d   conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
};
TORCH_MODULE(UNetBlock);

struct SpectrogramUNetImpl : torch::nn::Module {
    /// @param C  Number of sources to separate (default 2)
    SpectrogramUNetImpl(int64_t C = 2);

    /// @param mag  Magnitude spectrogram [B, 1, F, T_frames]
    /// @return     Masks [B, C, F, T_frames] in [0, 1]
    torch::Tensor forward(torch::Tensor mag);

    int64_t C_;

    // Encoder
    UNetBlock enc1{nullptr}, enc2{nullptr}, enc3{nullptr}, enc4{nullptr};
    torch::nn::MaxPool2d pool{nullptr};

    // Bottleneck
    UNetBlock bottleneck{nullptr};

    // Decoder
    torch::nn::ConvTranspose2d up4{nullptr}, up3{nullptr}, up2{nullptr}, up1{nullptr};
    UNetBlock dec4{nullptr}, dec3{nullptr}, dec2{nullptr}, dec1{nullptr};

    // Output
    torch::nn::Conv2d out_conv{nullptr};
};
TORCH_MODULE(SpectrogramUNet);

}  // namespace model
