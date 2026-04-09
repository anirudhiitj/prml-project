// ─────────────────────────────────────────────────────────────────────────────
// unet.cpp — Spectrogram U-Net implementation
// ─────────────────────────────────────────────────────────────────────────────
#include "unet.h"

namespace model {

// ═══════════════════ U-Net Conv Block ════════════════════════════════════════

UNetBlockImpl::UNetBlockImpl(int64_t in_ch, int64_t out_ch) {
    conv1 = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)));
    bn1   = register_module("bn1", torch::nn::BatchNorm2d(out_ch));
    conv2 = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3).padding(1)));
    bn2   = register_module("bn2", torch::nn::BatchNorm2d(out_ch));
}

torch::Tensor UNetBlockImpl::forward(torch::Tensor x) {
    x = torch::relu(bn1(conv1(x)));
    x = torch::relu(bn2(conv2(x)));
    return x;
}

// ═══════════════════ Spectrogram U-Net ═══════════════════════════════════════

SpectrogramUNetImpl::SpectrogramUNetImpl(int64_t C) : C_(C) {
    // Encoder: 1 → 32 → 64 → 128 → 256
    enc1 = register_module("enc1", UNetBlock(1, 32));
    enc2 = register_module("enc2", UNetBlock(32, 64));
    enc3 = register_module("enc3", UNetBlock(64, 128));
    enc4 = register_module("enc4", UNetBlock(128, 256));
    pool = register_module("pool", torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(2)));

    // Bottleneck
    bottleneck = register_module("bottleneck", UNetBlock(256, 512));

    // Decoder (ConvTranspose2d for upsampling)
    up4 = register_module("up4",
        torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(512, 256, 2).stride(2)));
    dec4 = register_module("dec4", UNetBlock(512, 256));

    up3 = register_module("up3",
        torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(256, 128, 2).stride(2)));
    dec3 = register_module("dec3", UNetBlock(256, 128));

    up2 = register_module("up2",
        torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(128, 64, 2).stride(2)));
    dec2 = register_module("dec2", UNetBlock(128, 64));

    up1 = register_module("up1",
        torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(64, 32, 2).stride(2)));
    dec1 = register_module("dec1", UNetBlock(64, 32));

    // Output: 32 → C masks
    out_conv = register_module("out_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, C, 1)));
}

torch::Tensor SpectrogramUNetImpl::forward(torch::Tensor mag) {
    // mag: [B, 1, F, T]
    // Pad to nearest multiple of 16 for 4 pooling layers
    auto orig_F = mag.size(2);
    auto orig_T = mag.size(3);
    int64_t pad_F = (16 - orig_F % 16) % 16;
    int64_t pad_T = (16 - orig_T % 16) % 16;
    if (pad_F > 0 || pad_T > 0) {
        mag = torch::nn::functional::pad(mag,
            torch::nn::functional::PadFuncOptions({0, pad_T, 0, pad_F}));
    }

    // Encoder
    auto e1 = enc1(mag);        // [B, 32, F, T]
    auto e2 = enc2(pool(e1));   // [B, 64, F/2, T/2]
    auto e3 = enc3(pool(e2));   // [B, 128, F/4, T/4]
    auto e4 = enc4(pool(e3));   // [B, 256, F/8, T/8]

    // Bottleneck
    auto b = bottleneck(pool(e4)); // [B, 512, F/16, T/16]

    // Decoder with skip connections
    auto d4 = up4(b);
    // Handle size mismatch from pooling
    if (d4.size(2) != e4.size(2) || d4.size(3) != e4.size(3)) {
        d4 = torch::nn::functional::pad(d4,
            torch::nn::functional::PadFuncOptions(
                {0, e4.size(3)-d4.size(3), 0, e4.size(2)-d4.size(2)}));
    }
    d4 = dec4(torch::cat({d4, e4}, 1));

    auto d3 = up3(d4);
    if (d3.size(2) != e3.size(2) || d3.size(3) != e3.size(3))
        d3 = torch::nn::functional::pad(d3,
            torch::nn::functional::PadFuncOptions(
                {0, e3.size(3)-d3.size(3), 0, e3.size(2)-d3.size(2)}));
    d3 = dec3(torch::cat({d3, e3}, 1));

    auto d2 = up2(d3);
    if (d2.size(2) != e2.size(2) || d2.size(3) != e2.size(3))
        d2 = torch::nn::functional::pad(d2,
            torch::nn::functional::PadFuncOptions(
                {0, e2.size(3)-d2.size(3), 0, e2.size(2)-d2.size(2)}));
    d2 = dec2(torch::cat({d2, e2}, 1));

    auto d1 = up1(d2);
    if (d1.size(2) != e1.size(2) || d1.size(3) != e1.size(3))
        d1 = torch::nn::functional::pad(d1,
            torch::nn::functional::PadFuncOptions(
                {0, e1.size(3)-d1.size(3), 0, e1.size(2)-d1.size(2)}));
    d1 = dec1(torch::cat({d1, e1}, 1));

    // Output masks with sigmoid
    auto masks = torch::sigmoid(out_conv(d1));  // [B, C, F_padded, T_padded]

    // Trim padding
    masks = masks.index({torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::indexing::Slice(0, orig_F),
                         torch::indexing::Slice(0, orig_T)});

    return masks;  // [B, C, F, T] in [0, 1]
}

}  // namespace model
