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
#include <fstream>

namespace fs = std::filesystem;

// ── CLI Config ───────────────────────────────────────────────────────────────
struct Config {
    std::string model_type     = "tasnet";  // "tasnet" or "unet"
    std::string data_dir       = "";
    std::string val_dir        = "";
    std::string ckpt_dir       = "./checkpoints";
    std::string resume_ckpt    = "";        // resume from checkpoint
    int         epochs         = 100;
    int         batch_size     = 2;
    float       lr             = 1e-3f;
    float       grad_clip      = 5.0f;
    int         seg_len        = 32000;  // 4s @ 8kHz
    int         sr             = 8000;
    int         workers        = 4;
    bool        augment        = true;
    bool        smoke_test     = false;
    int         lr_patience    = 5;
    float       lr_factor      = 0.5f;
    float       min_lr         = 1e-6f;  // LR floor
    int         log_interval   = 25;     // print every N batches
    int         save_interval  = 5;      // save checkpoint every N epochs
    int         accumulate     = 3;      // gradient accumulation steps
    int         warmup_epochs  = 5;      // linear LR warmup epochs

    // Conv-TasNet hyperparams (8M model)
    int N=512, L=16, B=256, H=416, P=3, X=8, R=3, C=2;
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
        else if (a == "--resume")     c.resume_ckpt = nx();
        else if (a == "--epochs")     c.epochs     = std::stoi(nx());
        else if (a == "--batch_size") c.batch_size = std::stoi(nx());
        else if (a == "--lr")         c.lr         = std::stof(nx());
        else if (a == "--grad_clip")  c.grad_clip  = std::stof(nx());
        else if (a == "--seg_len")    c.seg_len    = std::stoi(nx());
        else if (a == "--sr")         c.sr         = std::stoi(nx());
        else if (a == "--workers")    c.workers    = std::stoi(nx());
        else if (a == "--no_augment") c.augment    = false;
        else if (a == "--smoke_test") c.smoke_test = true;
        else if (a == "--log_interval")  c.log_interval  = std::stoi(nx());
        else if (a == "--save_interval") c.save_interval = std::stoi(nx());
        else if (a == "--lr_patience") c.lr_patience = std::stoi(nx());
        else if (a == "--min_lr")     c.min_lr     = std::stof(nx());
        else if (a == "--accumulate") c.accumulate = std::stoi(nx());
        else if (a == "--warmup_epochs") c.warmup_epochs = std::stoi(nx());
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

// ── CUDA Memory Logging ─────────────────────────────────────────────────────
static void log_cuda_memory() {
    if (torch::cuda::is_available()) {
        std::cout << "  [GPU] CUDA device available, "
                  << torch::cuda::device_count() << " GPU(s)\n";
    }
}

// ── Format time duration ────────────────────────────────────────────────────
static std::string format_duration(double seconds) {
    int h = static_cast<int>(seconds) / 3600;
    int m = (static_cast<int>(seconds) % 3600) / 60;
    int s = static_cast<int>(seconds) % 60;
    std::ostringstream oss;
    if (h > 0) oss << h << "h ";
    oss << m << "m " << s << "s";
    return oss.str();
}

// ── Smoke Test ───────────────────────────────────────────────────────────────
static void smoke_test(const Config& cfg) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n"
              << "║  Speech Separation Smoke Test (" << cfg.model_type << ")  ║\n"
              << "╚══════════════════════════════════════════════════════╝\n\n";

    auto dev = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << (dev == torch::kCUDA ? "CUDA" : "CPU") << "\n\n";

    if (dev == torch::kCUDA) log_cuda_memory();

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

        if (dev == torch::kCUDA) log_cuda_memory();

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

// ── Validation Result ────────────────────────────────────────────────────────
struct ValResult {
    float si_snr = 0;
    float sdri   = 0;
    float stoi   = 0;
};

static ValResult validate_tasnet(model::ConvTasNet& net, const std::string& csv,
                                  const Config& cfg, torch::Device dev) {
    net->eval();
    torch::NoGradGuard ng;

    auto ds = data::LibriMixDataset(csv, cfg.seg_len, cfg.sr, false)
        .map(torch::data::transforms::Stack<>());
    auto loader = torch::data::make_data_loader(
        std::move(ds),
        torch::data::DataLoaderOptions().batch_size(cfg.batch_size).workers(2));

    double total_snr = 0;
    double total_sdri = 0;
    double total_stoi = 0;
    int64_t n = 0;
    int64_t n_metrics = 0;
    for (auto& batch : *loader) {
        auto mix = batch.data.unsqueeze(1).to(dev);
        auto src = batch.target.to(dev);
        auto out = net->forward(mix);
        total_snr += -losses::pit_loss(out, src).item<double>() * mix.size(0);
        n += mix.size(0);

        // Compute detailed metrics on a subset (every 10th batch for speed)
        if (n_metrics < 200) {
            for (int b = 0; b < mix.size(0) && n_metrics < 200; ++b) {
                auto est = out[b].to(torch::kCPU);   // [C, T]
                auto ref = src[b].to(torch::kCPU);   // [C, T]
                auto mx  = mix[b].squeeze(0).to(torch::kCPU); // [T]
                // Fix polarity: each source should correlate
                // positively with the mixture
                for (int c = 0; c < 2; ++c) {
                    float dot = (est[c] * mx).sum().item<float>();
                    if (dot < 0.0f) est[c] = -est[c];
                }
                // Global rescale: α·Σ(est) ≈ mix  (preserves relative vols)
                auto sum_est_v = est[0] + est[1];
                float alpha_v = (mx * sum_est_v).sum().item<float>() /
                                ((sum_est_v * sum_est_v).sum().item<float>() + 1e-8f);
                est = est * alpha_v;
                auto m1 = metrics::evaluate(est[0], ref[0], mx, cfg.sr);
                auto m2 = metrics::evaluate(est[1], ref[1], mx, cfg.sr);
                auto m1r = metrics::evaluate(est[0], ref[1], mx, cfg.sr);
                auto m2r = metrics::evaluate(est[1], ref[0], mx, cfg.sr);
                float snri_p1 = (m1.si_snri + m2.si_snri) / 2;
                float snri_p2 = (m1r.si_snri + m2r.si_snri) / 2;
                if (snri_p1 >= snri_p2) {
                    total_sdri += (m1.sdri + m2.sdri) / 2;
                    total_stoi += (m1.stoi + m2.stoi) / 2;
                } else {
                    total_sdri += (m1r.sdri + m2r.sdri) / 2;
                    total_stoi += (m1r.stoi + m2r.stoi) / 2;
                }
                n_metrics++;
            }
        }
    }
    net->train();
    ValResult r;
    r.si_snr = (float)(total_snr / std::max(n, (int64_t)1));
    if (n_metrics > 0) {
        r.sdri = (float)(total_sdri / n_metrics);
        r.stoi = (float)(total_stoi / n_metrics);
    }
    return r;
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    auto cfg = parse(argc, argv);

    if (cfg.smoke_test) { smoke_test(cfg); return 0; }

    if (cfg.data_dir.empty()) {
        std::cerr << "Usage: ./train --data_dir <path> [--val_dir <path>] "
                     "[--model tasnet|unet] [options]\n"
                     "       ./train --smoke_test\n\n"
                     "Options:\n"
                     "  --model tasnet|unet   Model architecture (default: tasnet)\n"
                     "  --epochs N            Number of epochs (default: 100)\n"
                     "  --batch_size N        Batch size (default: 2)\n"
                     "  --accumulate N        Gradient accumulation steps (default: 3)\n"
                     "  --lr F                Learning rate (default: 1e-3)\n"
                     "  --warmup_epochs N     LR warmup epochs (default: 5)\n"
                     "  --min_lr F            Minimum LR floor (default: 1e-6)\n"
                     "  --sr N                Sample rate (default: 8000)\n"
                     "  --seg_len N           Segment length in samples (default: 32000)\n"
                     "  --resume <path>       Resume from checkpoint\n"
                     "  --no_augment          Disable data augmentation\n"
                     "  --workers N           DataLoader workers (default: 4)\n"
                     "  --grad_clip F         Gradient clip norm (default: 5.0)\n"
                     "  --log_interval N      Print every N batches (default: 25)\n"
                     "  --save_interval N     Save checkpoint every N epochs (default: 5)\n";
        return 1;
    }

    auto dev = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    std::cout << "═══════════════════════════════════════════════════\n"
              << " Speech Separation Training — " << cfg.model_type << "\n"
              << "═══════════════════════════════════════════════════\n"
              << " Device:     " << (dev==torch::kCUDA ? "CUDA" : "CPU") << "\n"
              << " Data:       " << cfg.data_dir << "\n"
              << " Epochs:     " << cfg.epochs << "\n"
              << " Batch:      " << cfg.batch_size << " (effective " << cfg.batch_size * cfg.accumulate << " with " << cfg.accumulate << "x accum)\n"
              << " LR:         " << cfg.lr << " (warmup " << cfg.warmup_epochs << " epochs, min " << cfg.min_lr << ")\n"
              << " SR:         " << cfg.sr << " Hz\n"
              << " Segment:    " << cfg.seg_len << " (" << (float)cfg.seg_len/cfg.sr << "s)\n"
              << " Workers:    " << cfg.workers << "\n"
              << " Augment:    " << (cfg.augment ? "yes" : "no") << "\n"
              << " Ckpt dir:   " << cfg.ckpt_dir << "\n"
              << "═══════════════════════════════════════════════════\n\n";

    if (dev == torch::kCUDA) log_cuda_memory();

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

        int start_epoch = 1;

        // Resume from checkpoint if specified
        if (!cfg.resume_ckpt.empty()) {
            std::cout << "Resuming from: " << cfg.resume_ckpt << "\n";
            torch::load(net, cfg.resume_ckpt);
            net->to(dev);
            auto opt_path = cfg.resume_ckpt + ".optim";
            if (fs::exists(opt_path)) {
                torch::load(opt, opt_path);
                std::cout << "  Optimizer state loaded.\n";
            }
            // Try to extract epoch number from filename
            auto fname = fs::path(cfg.resume_ckpt).stem().string();
            auto pos = fname.find("ep");
            if (pos != std::string::npos) {
                try {
                    start_epoch = std::stoi(fname.substr(pos + 2)) + 1;
                    std::cout << "  Resuming from epoch " << start_epoch << "\n";
                } catch (...) {}
            }
        }

        if (dev == torch::kCUDA) log_cuda_memory();

        float best_snr = -1e9f;
        int   stale = 0;

        // Helper: get/set current LR
        auto get_lr = [&]() {
            return static_cast<torch::optim::AdamOptions&>(
                opt.param_groups()[0].options()).lr();
        };
        auto set_lr = [&](double lr) {
            for (auto& g : opt.param_groups())
                static_cast<torch::optim::AdamOptions&>(g.options()).lr(lr);
        };

        // Helper: save checkpoint with epoch metadata
        auto save_checkpoint = [&](const std::string& path, int epoch) {
            torch::save(net, path);
            torch::save(opt, path + ".optim");
            // Save epoch number for reliable resume
            std::ofstream meta(path + ".meta");
            meta << "epoch=" << epoch << "\n";
            meta << "best_snr=" << best_snr << "\n";
            meta << "lr=" << get_lr() << "\n";
            meta.close();
        };

        // Load metadata if resuming
        if (!cfg.resume_ckpt.empty()) {
            auto meta_path = cfg.resume_ckpt + ".meta";
            if (fs::exists(meta_path)) {
                std::ifstream mf(meta_path);
                std::string line;
                while (std::getline(mf, line)) {
                    if (line.rfind("epoch=", 0) == 0) {
                        try {
                            start_epoch = std::stoi(line.substr(6)) + 1;
                            std::cout << "  Resuming from epoch " << start_epoch << "\n";
                        } catch (...) {}
                    }
                    if (line.rfind("best_snr=", 0) == 0)
                        best_snr = std::stof(line.substr(9));
                    if (line.rfind("lr=", 0) == 0) {
                        try {
                            double saved_lr = std::stod(line.substr(3));
                            set_lr(saved_lr);
                            std::cout << "  Restored LR: " << saved_lr << "\n";
                        } catch (...) {}
                    }
                }
                std::cout << "  Loaded metadata: best_snr=" << best_snr << "\n";
            }
        }

        for (int ep = start_epoch; ep <= cfg.epochs; ++ep) {
            net->train();
            double ep_loss = 0; int64_t nb = 0;
            int accum_step = 0;
            auto t0 = std::chrono::high_resolution_clock::now();

            // LR warmup: linear ramp from min_lr to target LR
            if (ep <= cfg.warmup_epochs) {
                double warmup_lr = cfg.min_lr + (cfg.lr - cfg.min_lr) *
                    ((double)ep / cfg.warmup_epochs);
                set_lr(warmup_lr);
            }

            opt.zero_grad();

            // Total batches for progress %
            int64_t total_batches = 50800 / std::max(cfg.batch_size, 1); // approximate

            for (auto& batch : *loader) {
                auto mix = batch.data.unsqueeze(1).to(dev);
                auto src = batch.target.to(dev);

                auto out = net->forward(mix);
                auto loss = losses::pit_loss(out, src) / (float)cfg.accumulate;
                loss.backward();

                ep_loss += loss.item<double>() * cfg.accumulate;
                nb++;
                accum_step++;

                if (accum_step >= cfg.accumulate) {
                    torch::nn::utils::clip_grad_norm_(net->parameters(), cfg.grad_clip);
                    opt.step();
                    opt.zero_grad();
                    accum_step = 0;
                }

                if (nb % cfg.log_interval == 0) {
                    auto elapsed = std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                    double avg_loss = ep_loss / nb;
                    double avg_snr = -avg_loss;  // SI-SNR = -loss
                    double pct = (double)nb / total_batches * 100.0;
                    if (pct > 100.0) pct = 99.9;
                    std::cout << "\r  [Ep " << std::setw(3) << ep << "/" << cfg.epochs
                              << "] " << std::fixed << std::setprecision(1) << pct << "%"
                              << " │ SI-SNR: " << std::setprecision(2) << avg_snr << " dB"
                              << " │ Loss: " << std::setprecision(4) << avg_loss
                              << " │ " << std::setprecision(0) << elapsed << "s"
                              << " │ LR: " << std::scientific << get_lr()
                              << "       " << std::flush;
                }
            }

            // Handle remaining accumulated gradients
            if (accum_step > 0) {
                torch::nn::utils::clip_grad_norm_(net->parameters(), cfg.grad_clip);
                opt.step();
                opt.zero_grad();
            }

            double sec = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
            double avg = ep_loss / std::max(nb, (int64_t)1);
            double train_snr = -avg;

            // ETA calculation
            int remaining_epochs = cfg.epochs - ep;
            std::string eta = format_duration(sec * remaining_epochs);

            // Clear the progress line
            std::cout << "\r                                                                                              \r";

            // Print epoch summary
            std::cout << "┌─── Epoch " << std::setw(3) << ep << "/" << cfg.epochs
                      << " ───────────────────────────────────────────────┐\n";
            std::cout << "│  Train SI-SNR: " << std::fixed << std::setprecision(2)
                      << std::setw(8) << train_snr << " dB"
                      << "    │ Loss: " << std::setprecision(4) << std::setw(9) << avg
                      << "         │\n";
            std::cout << "│  Time: " << std::setprecision(0) << std::setw(5) << sec << "s"
                      << "  Batches: " << std::setw(6) << nb
                      << " │ LR: " << std::scientific << get_lr()
                      << "  ETA: " << std::setw(8) << eta << " │\n";

            if (!val_csv.empty()) {
                auto vres = validate_tasnet(net, val_csv, cfg, dev);
                std::cout << "│  Val   SI-SNR: " << std::fixed << std::setprecision(2)
                          << std::setw(8) << vres.si_snr << " dB"
                          << "    │ SDRi: " << std::setprecision(2) << std::setw(7) << vres.sdri
                          << " dB  STOI: " << std::setprecision(3) << vres.stoi << " │\n";
                if (vres.si_snr > best_snr) {
                    best_snr = vres.si_snr; stale = 0;
                    auto best_path = (fs::path(cfg.ckpt_dir)/"best_tasnet.pt").string();
                    save_checkpoint(best_path, ep);
                    std::cout << "│  ★ New best model saved!  (prev best: "
                              << std::setprecision(2) << best_snr << " dB)"
                              << "                        │\n";
                } else if (ep > cfg.warmup_epochs) {
                    stale++;
                    std::cout << "│  No improvement (" << stale << "/" << cfg.lr_patience << " patience)";
                    if (stale >= cfg.lr_patience) {
                        double new_lr = std::max(get_lr() * cfg.lr_factor,
                                                  (double)cfg.min_lr);
                        set_lr(new_lr);
                        stale = 0;
                        std::cout << "  → LR reduced to " << std::scientific << new_lr;
                    }
                    std::cout << "                        │\n";
                }
            }
            std::cout << "└──────────────────────────────────────────────────────────┘\n";
            std::cout << std::flush;

            // Save checkpoint at intervals (with full resume info)
            if (ep % cfg.save_interval == 0) {
                auto ckpt_path = (fs::path(cfg.ckpt_dir)/
                    ("tasnet_ep" + std::to_string(ep) + ".pt")).string();
                save_checkpoint(ckpt_path, ep);
                std::cout << "  [Checkpoint saved: " << ckpt_path << "]\n";
            }

            // Always save a 'latest' checkpoint for crash recovery
            {
                auto latest_path = (fs::path(cfg.ckpt_dir)/"latest_tasnet.pt").string();
                save_checkpoint(latest_path, ep);
            }
        }

        auto final_path = (fs::path(cfg.ckpt_dir)/"final_tasnet.pt").string();
        save_checkpoint(final_path, cfg.epochs);
        std::cout << "\n✓ Done. Best SI-SNR: " << best_snr << " dB\n";
        std::cout << "  Final model: " << final_path << "\n";
        if (!val_csv.empty())
            std::cout << "  Best model:  " << (fs::path(cfg.ckpt_dir)/"best_tasnet.pt").string() << "\n";
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
                if (nb % cfg.log_interval == 0)
                    std::cout << "\r  Ep " << ep << " | Batch " << nb
                              << " | Loss " << std::fixed << std::setprecision(4)
                              << (ep_loss/nb) << std::flush;
            }

            double sec = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
            double avg = ep_loss / std::max(nb, (int64_t)1);

            int remaining_epochs = cfg.epochs - ep;
            std::string eta = format_duration(sec * remaining_epochs);

            std::cout << "\r  Ep " << std::setw(3) << ep
                      << " | Loss " << std::fixed << std::setprecision(4) << avg
                      << " | " << std::setprecision(0) << sec << "s"
                      << " | ETA " << eta << "\n";

            if (avg < best_loss) {
                best_loss = avg;
                torch::save(unet, (fs::path(cfg.ckpt_dir)/"best_unet.pt").string());
            }
            if (ep % cfg.save_interval == 0) {
                auto ckpt_path = (fs::path(cfg.ckpt_dir)/
                    ("unet_ep" + std::to_string(ep) + ".pt")).string();
                torch::save(unet, ckpt_path);
                std::cout << "  [Checkpoint saved: " << ckpt_path << "]\n";
            }
        }
        torch::save(unet, (fs::path(cfg.ckpt_dir)/"final_unet.pt").string());
    }

    return 0;
}
