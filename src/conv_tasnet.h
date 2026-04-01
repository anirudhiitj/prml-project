#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// conv_tasnet.h — Conv-TasNet (Luo & Mesgarani, IEEE/ACM TASLP 2019)
//
// Time-domain audio separation network:
//   Encoder (1D Conv) → TCN Mask Network → Decoder (Transposed 1D Conv)
//
// Default config: N=256, L=16, B=128, H=256, P=3, X=8, R=3, C=2
// Parameters: ~5.1M
// ─────────────────────────────────────────────────────────────────────────────
#include <torch/torch.h>

namespace model {

// ── Layer Normalizations ─────────────────────────────────────────────────────

struct GlobalLayerNormImpl : torch::nn::Module {
    GlobalLayerNormImpl(int64_t ch, float eps = 1e-8);
    torch::Tensor forward(torch::Tensor x);  // [B, N, T]
    torch::Tensor gamma_, beta_;
    float eps_;
};
TORCH_MODULE(GlobalLayerNorm);

struct ChannelLayerNormImpl : torch::nn::Module {
    ChannelLayerNormImpl(int64_t ch, float eps = 1e-8);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor gamma_, beta_;
    float eps_;
};
TORCH_MODULE(ChannelLayerNorm);

// ── Depth-wise Separable Conv Block ──────────────────────────────────────────

struct DepthSepBlockImpl : torch::nn::Module {
    DepthSepBlockImpl(int64_t B, int64_t H, int64_t P, int64_t dilation);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv1d  conv_in{nullptr}, dconv{nullptr};
    torch::nn::Conv1d  conv_skip{nullptr}, conv_res{nullptr};
    torch::nn::PReLU   prelu1{nullptr}, prelu2{nullptr};
    ChannelLayerNorm   norm1{nullptr}, norm2{nullptr};
};
TORCH_MODULE(DepthSepBlock);

// ── TCN Separation Module ────────────────────────────────────────────────────

struct TCNImpl : torch::nn::Module {
    TCNImpl(int64_t N, int64_t B, int64_t H, int64_t P,
            int64_t X, int64_t R, int64_t C);
    torch::Tensor forward(torch::Tensor enc_out);  // [B,N,L] → [B,C,N,L]

    GlobalLayerNorm     ln{nullptr};
    torch::nn::Conv1d   bottleneck{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::Conv1d   mask_conv{nullptr};
    int64_t N_, C_;
};
TORCH_MODULE(TCN);

// ── Full Conv-TasNet Model ───────────────────────────────────────────────────

struct ConvTasNetImpl : torch::nn::Module {
    ConvTasNetImpl(int64_t N=256, int64_t L=16, int64_t B=128,
                   int64_t H=256, int64_t P=3,  int64_t X=8,
                   int64_t R=3,   int64_t C=2,  int sr=8000);

    /// @param mixture [B, T] or [B, 1, T]
    /// @return        [B, C, T] separated sources
    torch::Tensor forward(torch::Tensor mixture);

    int64_t N_, L_, stride_, C_;
    int sr_;
    torch::nn::Conv1d          encoder{nullptr};
    TCN                        separator{nullptr};
    torch::nn::ConvTranspose1d decoder{nullptr};
};
TORCH_MODULE(ConvTasNet);

}  // namespace model
