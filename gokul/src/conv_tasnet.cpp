// ─────────────────────────────────────────────────────────────────────────────
// conv_tasnet.cpp — Conv-TasNet implementation
// ─────────────────────────────────────────────────────────────────────────────
#include "conv_tasnet.h"
#include <cmath>

namespace model {

// ═══════════════════ Global Layer Normalization ══════════════════════════════

GlobalLayerNormImpl::GlobalLayerNormImpl(int64_t ch, float eps) : eps_(eps) {
    gamma_ = register_parameter("gamma", torch::ones({1, ch, 1}));
    beta_  = register_parameter("beta",  torch::zeros({1, ch, 1}));
}

torch::Tensor GlobalLayerNormImpl::forward(torch::Tensor x) {
    auto mean = x.mean({1, 2}, true);
    auto var  = x.var({1, 2}, false, true);
    return gamma_ * (x - mean) / (var + eps_).sqrt() + beta_;
}

// ═══════════════════ Channel Layer Normalization ═════════════════════════════

ChannelLayerNormImpl::ChannelLayerNormImpl(int64_t ch, float eps) : eps_(eps) {
    gamma_ = register_parameter("gamma", torch::ones({1, ch, 1}));
    beta_  = register_parameter("beta",  torch::zeros({1, ch, 1}));
}

torch::Tensor ChannelLayerNormImpl::forward(torch::Tensor x) {
    auto mean = x.mean(1, true);
    auto var  = x.var(1, false, true);
    return gamma_ * (x - mean) / (var + eps_).sqrt() + beta_;
}

// ═══════════════════ Depth-wise Separable Conv Block ═════════════════════════

DepthSepBlockImpl::DepthSepBlockImpl(int64_t B, int64_t H, int64_t P,
                                     int64_t dilation) {
    int64_t pad = dilation * (P - 1) / 2;

    conv_in = register_module("conv_in",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(B, H, 1)));
    prelu1  = register_module("prelu1",
        torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(H)));
    norm1   = register_module("norm1", ChannelLayerNorm(H));

    dconv = register_module("dconv",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(H, H, P)
            .dilation(dilation).padding(pad).groups(H)));
    prelu2 = register_module("prelu2",
        torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(H)));
    norm2  = register_module("norm2", ChannelLayerNorm(H));

    conv_skip = register_module("conv_skip",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(H, B, 1)));
    conv_res  = register_module("conv_res",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(H, B, 1)));
}

torch::Tensor DepthSepBlockImpl::forward(torch::Tensor x) {
    auto h = norm1(prelu1(conv_in(x)));
    h = norm2(prelu2(dconv(h)));
    return x + conv_res(h);  // residual (skip captured externally in TCN)
}

// ═══════════════════ TCN ═════════════════════════════════════════════════════

TCNImpl::TCNImpl(int64_t N, int64_t B, int64_t H, int64_t P,
                 int64_t X, int64_t R, int64_t C)
    : N_(N), C_(C) {
    ln = register_module("ln", GlobalLayerNorm(N));
    bottleneck = register_module("bottleneck",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(N, B, 1)));
    blocks = register_module("blocks", torch::nn::ModuleList());
    for (int64_t r = 0; r < R; ++r)
        for (int64_t x = 0; x < X; ++x)
            blocks->push_back(DepthSepBlock(B, H, P,
                static_cast<int64_t>(std::pow(2, x))));
    mask_conv = register_module("mask_conv",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(B, C * N, 1)));
}

torch::Tensor TCNImpl::forward(torch::Tensor enc_out) {
    auto B = enc_out.size(0);
    auto L = enc_out.size(2);
    auto x = bottleneck(ln(enc_out));             // [B, Bdim, L]
    torch::Tensor skip_sum = torch::zeros_like(x);

    for (size_t i = 0; i < blocks->size(); ++i) {
        auto blk = blocks->ptr<DepthSepBlockImpl>(i);
        auto h = blk->norm1(blk->prelu1(blk->conv_in(x)));
        h = blk->norm2(blk->prelu2(blk->dconv(h)));
        skip_sum = skip_sum + blk->conv_skip(h);
        x = x + blk->conv_res(h);
    }

    auto masks = mask_conv(skip_sum).view({B, C_, N_, L});
    masks = torch::relu(masks);
    return enc_out.unsqueeze(1) * masks;  // [B, C, N, L]
}

// ═══════════════════ Conv-TasNet ═════════════════════════════════════════════

ConvTasNetImpl::ConvTasNetImpl(int64_t N, int64_t L, int64_t B,
                               int64_t H, int64_t P, int64_t X,
                               int64_t R, int64_t C, int sr)
    : N_(N), L_(L), stride_(L/2), C_(C), sr_(sr) {
    encoder = register_module("encoder",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(1, N, L)
            .stride(stride_).bias(false)));
    separator = register_module("separator", TCN(N, B, H, P, X, R, C));
    decoder = register_module("decoder",
        torch::nn::ConvTranspose1d(
            torch::nn::ConvTranspose1dOptions(N, 1, L)
                .stride(stride_).bias(false)));
}

torch::Tensor ConvTasNetImpl::forward(torch::Tensor mix) {
    if (mix.dim() == 2) mix = mix.unsqueeze(1);  // [B,1,T]
    auto orig_len = mix.size(2);

    // Pad to multiple of stride
    int64_t rem = orig_len % stride_;
    if (rem > 0) mix = torch::nn::functional::pad(mix,
        torch::nn::functional::PadFuncOptions({0, stride_ - rem}));

    auto enc = torch::relu(encoder(mix));         // [B, N, L]
    auto masked = separator(enc);                 // [B, C, N, L]
    auto batch = masked.size(0);
    auto L_enc = masked.size(3);

    auto flat = masked.view({batch * C_, N_, L_enc});
    auto dec  = decoder(flat).view({batch, C_, -1});

    return dec.index({torch::indexing::Slice(), torch::indexing::Slice(),
                      torch::indexing::Slice(0, orig_len)});
}

}  // namespace model
