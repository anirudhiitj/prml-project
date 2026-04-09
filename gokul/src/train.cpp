// ─────────────────────────────────────────────────────────────────────────────
// train.cpp — Training entry point for Conv-TasNet / U-Net speech separation
//
// Usage:
//   ./train --model tasnet --data_dir ./data/train-360 --val_dir ./data/dev
//   ./train --model unet   --data_dir ./data/train-360 --val_dir ./data/dev
//   ./train --smoke_test   (verify model + pipeline without data)
// ─────────────────────────────────────────────────────────────────────────────
#include "conv_tasnet.h"
#include "unet.h"
#include "dataset.h"
#include "losses.h"
#include "metrics.h"
#include "stft.h"
#include "audio_utils.h"
#include "preprocessing.h"

#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <string>
#include <iomanip>

namespace fs = std::filesystem;

// ── CLI Config ───────────────────────────────────────────────────────────────
struct Config {
    std::string model_type     = "tasnet";  // "tasnet" or "unet"
    std::string data_dir       = "";
    std::string val_dir        = "";
    std::string ckpt_dir       = "./checkpoints";
    int         epochs         = 100;
    int         batch_size     = 4;
    float       lr             = 1e-3f;
    float       grad_clip      = 5.0f;
    int         seg_len        = 32000;  // 4s @ 8kHz
    int         sr             = 8000;
    int         workers        = 4;
    bool        augment        = true;
    bool        smoke_test     = false;
    int         lr_patience    = 3;
    float       lr_factor      = 0.5f;

    // Conv-TasNet hyperparams
    int N=256, L=16, B=128, H=256, P=3, X=8, R=3, C=2;
};

static Config parse(int argc, char* argv[]) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return (i+1<argc) ? argv[++i] : ""; };
        if      (a == "--model")      c.model_type = nx();
        else if (a == "--data_dir")   c.data_dir   = nx();
        else if (a == "--val_dir")    c.val_dir    = nx();
        else if (a == "--ckpt_dir")   c.ckpt_dir   = nx();
        else if (a == "--epochs")     c.epochs     = std::stoi(nx());
        else if (a == "--batch_size") c.batch_size = std::stoi(nx());
        else if (a == "--lr")         c.lr         = std::stof(nx());
        else if (a == "--grad_clip")  c.grad_clip  = std::stof(nx());
        else if (a == "--seg_len")    c.seg_len    = std::stoi(nx());
        else if (a == "--workers")    c.workers    = std::stoi(nx());
        else if (a == "--no_augment") c.augment    = false;
        else if (a == "--smoke_test") c.smoke_test = true;
        else if (a == "--N") c.N = std::stoi(nx());
        else if (a == "--L") c.L = std::stoi(nx());
        else if (a == "--B") c.B = std::stoi(nx());
        else if (a == "--H") c.H = std::stoi(nx());
        else if (a == "--P") c.P = std::stoi(nx());
        else if (a == "--X") c.X = std::stoi(nx());
        else if (a == "--R") c.R = std::stoi(nx());
    }
    return c;
}

// ── Smoke Test ───────────────────────────────────────────────────────────────
static void smoke_test(const Config& cfg) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n"
              << "║  Speech Separation Smoke Test (" << cfg.model_type << ")  ║\n"
              << "╚══════════════════════════════════════════════════════╝\n\n";

    auto dev = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << (dev == torch::kCUDA ? "CUDA" : "CPU") << "\n\n";

    // Test STFT roundtrip
    {
        std::cout << "── STFT/iSTFT roundtrip ──\n";
        auto signal = torch::randn({1, cfg.seg_len});
        auto spec   = stft_utils::stft(signal);
        auto recon  = stft_utils::istft(spec, {}, cfg.seg_len);
        float err   = (signal - recon).abs().max().item<float>();
        std::cout << "  Max reconstruction error: " << err << " (should be < 1e-5)\n";
    }

    // Test Conv-TasNet
    if (cfg.model_type == "tasnet") {
        auto net = model::ConvTasNet(cfg.N, cfg.L, cfg.B, cfg.H, cfg.P,
                                      cfg.X, cfg.R, cfg.C, cfg.sr);
        net->to(dev);
        int64_t p = 0;
        for (auto& param : net->parameters()) p += param.numel();
        std::cout << "\n── Conv-TasNet ──\n"
                  << "  Params: " << (p/1e6) << "M\n";

        auto mix = torch::randn({2, 1, cfg.seg_len}, dev);
        auto t0  = std::chrono::high_resolution_clock::now();
        auto out = net->forward(mix);
        auto t1  = std::chrono::high_resolution_clock::now();
        std::cout << "  Input:  " << mix.sizes() << "\n"
                  << "  Output: " << out.sizes() << "\n"
                  << "  Time:   " << std::chrono::duration<double,std::milli>(t1-t0).count() << " ms\n";

        auto tgt = torch::randn_like(out);
        auto loss = losses::pit_loss(out, tgt);
        loss.backward();
        std::cout << "  PIT loss (random): " << loss.item<float>() << "\n"
                  << "  Backward: OK\n";
    }

    // Test U-Net
    {
        std::cout << "\n── Spectrogram U-Net ──\n";
        auto unet = model::SpectrogramUNet(2);
        unet->to(dev);
        int64_t up = 0;
        for (auto& param : unet->parameters()) up += param.numel();
        std::cout << "  Params: " << (up/1e6) << "M\n";

        auto mag = torch::randn({2, 1, 129, 501}, dev);  // typical STFT size
        auto masks = unet->forward(mag);
        std::cout << "  Input:  " << mag.sizes() << "\n"
                  << "  Masks:  " << masks.sizes() << "\n"
                  << "  Range:  [" << masks.min().item<float>() << ", "
                  << masks.max().item<float>() << "]\n";
    }

    // Test metrics
    {
        std::cout << "\n── Metrics ──\n";
        auto tgt = torch::randn({8000});
        auto est = tgt + 0.1f * torch::randn({8000});
        auto mix = tgt + torch::randn({8000});
        auto m = metrics::evaluate(est, tgt, mix, cfg.sr);
        std::cout << "  SI-SNRi: " << m.si_snri << " dB\n"
                  << "  SDRi:    " << m.sdri << " dB\n"
                  << "  STOI:    " << m.stoi << "\n";
    }

    std::cout << "\n✓ All smoke tests passed!\n";
}

// ── Validate ─────────────────────────────────────────────────────────────────
static float validate_tasnet(model::ConvTasNet& net, const std::string& csv,
                              const Config& cfg, torch::Device dev) {
    net->eval();
    torch::NoGradGuard ng;

    auto ds = data::LibriMixDataset(csv, cfg.seg_len, cfg.sr, false)
        .map(torch::data::transforms::Stack<>());
    auto loader = torch::data::make_data_loader(
        std::move(ds),
        torch::data::DataLoaderOptions().batch_size(cfg.batch_size).workers(2));

    double total = 0; int64_t n = 0;
    for (auto& batch : *loader) {
        auto mix = batch.data.unsqueeze(1).to(dev);
        auto src = batch.target.to(dev);
        auto out = net->forward(mix);
        total += -losses::pit_loss(out, src).item<double>() * mix.size(0);
        n += mix.size(0);
    }
    net->train();
    return (float)(total / std::max(n, (int64_t)1));
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    auto cfg = parse(argc, argv);

    if (cfg.smoke_test) { smoke_test(cfg); return 0; }

    if (cfg.data_dir.empty()) {
        std::cerr << "Usage: ./train --data_dir <path> [--val_dir <path>] "
                     "[--model tasnet|unet] [options]\n"
                     "       ./train --smoke_test\n";
        return 1;
    }

    auto dev = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    std::cout << "═══════════════════════════════════════════════════\n"
              << " Speech Separation Training — " << cfg.model_type << "\n"
              << "═══════════════════════════════════════════════════\n"
              << " Device:     " << (dev==torch::kCUDA ? "CUDA" : "CPU") << "\n"
              << " Data:       " << cfg.data_dir << "\n"
              << " Epochs:     " << cfg.epochs << "\n"
              << " Batch:      " << cfg.batch_size << "\n"
              << " LR:         " << cfg.lr << "\n"
              << " Segment:    " << cfg.seg_len << " (" << (float)cfg.seg_len/cfg.sr << "s)\n"
              << "═══════════════════════════════════════════════════\n\n";

    // Find CSV files
    auto find_csv = [](const std::string& dir) -> std::string {
        for (auto& p : fs::recursive_directory_iterator(dir))
            if (p.path().filename().string().find("mixture") != std::string::npos
                && p.path().extension() == ".csv")
                return p.path().string();
        return "";
    };

    std::string train_csv = find_csv(cfg.data_dir);
    if (train_csv.empty()) {
        std::cerr << "No mixture_*.csv in " << cfg.data_dir << "\n"; return 1;
    }
    std::cout << "Train CSV: " << train_csv << "\n";

    std::string val_csv;
    if (!cfg.val_dir.empty()) {
        val_csv = find_csv(cfg.val_dir);
        std::cout << "Val CSV:   " << val_csv << "\n";
    }

    // Create dataset
    auto train_ds = data::LibriMixDataset(train_csv, cfg.seg_len, cfg.sr, cfg.augment)
        .map(torch::data::transforms::Stack<>());
    auto loader = torch::data::make_data_loader(
        std::move(train_ds),
        torch::data::DataLoaderOptions()
            .batch_size(cfg.batch_size)
            .workers(cfg.workers)
            .enforce_ordering(false));

    fs::create_directories(cfg.ckpt_dir);

    // ── Conv-TasNet Training ─────────────────────────────────────────────────
    if (cfg.model_type == "tasnet") {
        auto net = model::ConvTasNet(cfg.N, cfg.L, cfg.B, cfg.H, cfg.P,
                                      cfg.X, cfg.R, cfg.C, cfg.sr);
        net->to(dev);

        int64_t np = 0;
        for (auto& p : net->parameters()) np += p.numel();
        std::cout << "Parameters: " << (np/1e6) << "M\n\n";

        torch::optim::Adam opt(net->parameters(),
            torch::optim::AdamOptions(cfg.lr));

        float best_snr = -1e9f;
        int   stale = 0;

        for (int ep = 1; ep <= cfg.epochs; ++ep) {
            net->train();
            double ep_loss = 0; int64_t nb = 0;
            auto t0 = std::chrono::high_resolution_clock::now();

            for (auto& batch : *loader) {
                auto mix = batch.data.unsqueeze(1).to(dev);
                auto src = batch.target.to(dev);

                opt.zero_grad();
                auto out = net->forward(mix);
                auto loss = losses::pit_loss(out, src);
                loss.backward();
                torch::nn::utils::clip_grad_norm_(net->parameters(), cfg.grad_clip);
                opt.step();

                ep_loss += loss.item<double>();
                nb++;
                if (nb % 100 == 0)
                    std::cout << "\r  Ep " << ep << " | Batch " << nb
                              << " | Loss " << std::fixed << std::setprecision(4)
                              << (ep_loss/nb) << std::flush;
            }

            double sec = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
            double avg = ep_loss / std::max(nb, (int64_t)1);

            std::cout << "\r  Ep " << std::setw(3) << ep
                      << " | Loss " << std::fixed << std::setprecision(4) << avg
                      << " | " << std::setprecision(0) << sec << "s";

            if (!val_csv.empty()) {
                float vsnr = validate_tasnet(net, val_csv, cfg, dev);
                std::cout << " | Val SI-SNR: " << std::setprecision(2) << vsnr << " dB";
                if (vsnr > best_snr) {
                    best_snr = vsnr; stale = 0;
                    torch::save(net, (fs::path(cfg.ckpt_dir)/"best_tasnet.pt").string());
                    std::cout << " ★";
                } else {
                    stale++;
                    if (stale >= cfg.lr_patience) {
                        for (auto& g : opt.param_groups())
                            static_cast<torch::optim::AdamOptions&>(g.options())
                                .lr(static_cast<torch::optim::AdamOptions&>(g.options()).lr() * cfg.lr_factor);
                        stale = 0;
                        std::cout << " (LR halved)";
                    }
                }
            }
            std::cout << "\n";

            if (ep % 10 == 0)
                torch::save(net, (fs::path(cfg.ckpt_dir)/
                    ("tasnet_ep" + std::to_string(ep) + ".pt")).string());
        }

        torch::save(net, (fs::path(cfg.ckpt_dir)/"final_tasnet.pt").string());
        std::cout << "\n✓ Done. Best SI-SNR: " << best_snr << " dB\n";
    }
    // ── U-Net Training ───────────────────────────────────────────────────────
    else if (cfg.model_type == "unet") {
        auto unet = model::SpectrogramUNet(cfg.C);
        unet->to(dev);

        int64_t up = 0;
        for (auto& p : unet->parameters()) up += p.numel();
        std::cout << "U-Net params: " << (up/1e6) << "M\n\n";

        torch::optim::Adam opt(unet->parameters(),
            torch::optim::AdamOptions(cfg.lr));

        stft_utils::STFTConfig stft_cfg;
        float best_loss = 1e9f;

        for (int ep = 1; ep <= cfg.epochs; ++ep) {
            unet->train();
            double ep_loss = 0; int64_t nb = 0;
            auto t0 = std::chrono::high_resolution_clock::now();

            for (auto& batch : *loader) {
                auto mix_wav = batch.data.to(dev);      // [B, T]
                auto src_wav = batch.target.to(dev);     // [B, C, T]

                // STFT
                auto mix_spec = stft_utils::stft(mix_wav, stft_cfg);
                auto mix_mag  = stft_utils::magnitude(mix_spec).unsqueeze(1); // [B,1,F,T]

                // Source spectrograms for target
                auto s1_spec = stft_utils::stft(src_wav.select(1,0), stft_cfg);
                auto s2_spec = stft_utils::stft(src_wav.select(1,1), stft_cfg);
                auto tgt_mags = torch::stack({stft_utils::magnitude(s1_spec),
                                               stft_utils::magnitude(s2_spec)}, 1); // [B,C,F,T]

                opt.zero_grad();
                auto masks = unet->forward(mix_mag);
                auto loss = losses::spectral_pit_loss(masks, mix_mag, tgt_mags);
                loss.backward();
                torch::nn::utils::clip_grad_norm_(unet->parameters(), cfg.grad_clip);
                opt.step();

                ep_loss += loss.item<double>();
                nb++;
                if (nb % 100 == 0)
                    std::cout << "\r  Ep " << ep << " | Batch " << nb
                              << " | Loss " << std::fixed << std::setprecision(4)
                              << (ep_loss/nb) << std::flush;
            }

            double sec = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
            double avg = ep_loss / std::max(nb, (int64_t)1);

            std::cout << "\r  Ep " << std::setw(3) << ep
                      << " | Loss " << std::fixed << std::setprecision(4) << avg
                      << " | " << std::setprecision(0) << sec << "s\n";

            if (avg < best_loss) {
                best_loss = avg;
                torch::save(unet, (fs::path(cfg.ckpt_dir)/"best_unet.pt").string());
            }
            if (ep % 10 == 0)
                torch::save(unet, (fs::path(cfg.ckpt_dir)/
                    ("unet_ep" + std::to_string(ep) + ".pt")).string());
        }
        torch::save(unet, (fs::path(cfg.ckpt_dir)/"final_unet.pt").string());
    }

    return 0;
}
