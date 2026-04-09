// ─────────────────────────────────────────────────────────────────────────────
// inference.cpp — Separate a mixture WAV into individual speaker tracks
//
// Usage:
//   ./inference --model tasnet --checkpoint best_tasnet.pt --input mix.wav
//   ./inference --model unet   --checkpoint best_unet.pt   --input mix.wav
//   ./inference --smoke_test
//
// Optional: --ref_s1 clean1.wav --ref_s2 clean2.wav  (evaluates metrics)
// ─────────────────────────────────────────────────────────────────────────────
#include "conv_tasnet.h"
#include "unet.h"
#include "audio_utils.h"
#include "preprocessing.h"
#include "stft.h"
#include "losses.h"
#include "metrics.h"

#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <chrono>

namespace fs = std::filesystem;

struct InfConfig {
    std::string model_type  = "tasnet";
    std::string checkpoint  = "";
    std::string input_wav   = "";
    std::string output_dir  = "./output";
    std::string ref_s1      = "";
    std::string ref_s2      = "";
    int  sr                 = 8000;
    bool smoke_test         = false;
    int N=256, L=16, B=128, H=256, P=3, X=8, R=3, C=2;
};

static InfConfig parse(int argc, char* argv[]) {
    InfConfig c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return (i+1<argc) ? argv[++i] : ""; };
        if      (a == "--model")      c.model_type = nx();
        else if (a == "--checkpoint") c.checkpoint = nx();
        else if (a == "--input")      c.input_wav  = nx();
        else if (a == "--output_dir") c.output_dir = nx();
        else if (a == "--ref_s1")     c.ref_s1     = nx();
        else if (a == "--ref_s2")     c.ref_s2     = nx();
        else if (a == "--sr")         c.sr         = std::stoi(nx());
        else if (a == "--smoke_test") c.smoke_test = true;
    }
    return c;
}

static void run_smoke_test(const InfConfig& cfg) {
    std::cout << "\n═══════ Inference Smoke Test ═══════\n\n";
    auto dev = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << (dev == torch::kCUDA ? "CUDA" : "CPU") << "\n";

    // Conv-TasNet inference test
    {
        auto net = model::ConvTasNet(cfg.N, cfg.L, cfg.B, cfg.H, cfg.P,
                                      cfg.X, cfg.R, cfg.C, cfg.sr);
        net->to(dev);
        net->eval();
        auto mix = torch::randn({1, 1, 32000}, dev);
        auto out = net->forward(mix);
        std::cout << "Conv-TasNet: " << mix.sizes() << " → " << out.sizes() << "\n";
    }

    // U-Net inference test
    {
        auto unet = model::SpectrogramUNet(2);
        unet->to(dev);
        unet->eval();
        auto mag = torch::rand({1, 1, 129, 501}, dev) * 10.0f;
        auto masks = unet->forward(mag);
        std::cout << "U-Net:       masks " << masks.sizes()
                  << " range [" << masks.min().item<float>() << ", "
                  << masks.max().item<float>() << "]\n";
    }

    // Audio I/O roundtrip
    {
        auto sig = torch::randn({1, 8000});
        audio::save_wav("/tmp/smoke_test_sep.wav", sig, cfg.sr);
        auto loaded = audio::load_wav("/tmp/smoke_test_sep.wav", cfg.sr);
        float err = (sig - loaded.index({torch::indexing::Slice(),
                      torch::indexing::Slice(0, sig.size(1))})).abs().max().item<float>();
        std::cout << "Audio I/O roundtrip error: " << err << "\n";
    }

    std::cout << "\n✓ Inference smoke test passed!\n";
}

int main(int argc, char* argv[]) {
    auto cfg = parse(argc, argv);

    if (cfg.smoke_test) { run_smoke_test(cfg); return 0; }

    if (cfg.checkpoint.empty() || cfg.input_wav.empty()) {
        std::cerr << "Usage: ./inference --model tasnet|unet --checkpoint <path> "
                     "--input <wav> [--ref_s1 <wav> --ref_s2 <wav>]\n"
                     "       ./inference --smoke_test\n";
        return 1;
    }

    auto dev = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    fs::create_directories(cfg.output_dir);

    std::cout << "═══════════════════════════════════════════════════\n"
              << " Inference — " << cfg.model_type << "\n"
              << "═══════════════════════════════════════════════════\n";

    // Load mixture
    auto mix_wav = audio::load_wav(cfg.input_wav, cfg.sr);  // [1, T]
    std::cout << "Input: " << cfg.input_wav << " (" << mix_wav.size(1)
              << " samples, " << (float)mix_wav.size(1)/cfg.sr << "s)\n";

    // Preprocess
    mix_wav = preprocess::normalize(mix_wav);
    auto T = mix_wav.size(1);

    torch::Tensor separated;  // [C, T]

    if (cfg.model_type == "tasnet") {
        auto net = model::ConvTasNet(cfg.N, cfg.L, cfg.B, cfg.H, cfg.P,
                                      cfg.X, cfg.R, cfg.C, cfg.sr);
        torch::load(net, cfg.checkpoint);
        net->to(dev);
        net->eval();
        torch::NoGradGuard ng;

        auto input = mix_wav.unsqueeze(0).to(dev);  // [1, 1, T]
        auto t0 = std::chrono::high_resolution_clock::now();
        separated = net->forward(input).squeeze(0).to(torch::kCPU);  // [C, T]
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Conv-TasNet separation: "
                  << std::chrono::duration<double,std::milli>(t1-t0).count() << " ms\n";
    }
    else if (cfg.model_type == "unet") {
        auto unet = model::SpectrogramUNet(cfg.C);
        torch::load(unet, cfg.checkpoint);
        unet->to(dev);
        unet->eval();
        torch::NoGradGuard ng;

        stft_utils::STFTConfig stft_cfg;
        auto spec    = stft_utils::stft(mix_wav);
        auto mag     = stft_utils::magnitude(spec).unsqueeze(0).unsqueeze(0).to(dev);
        auto phi     = stft_utils::phase(spec);

        auto masks   = unet->forward(mag).squeeze(0).to(torch::kCPU);  // [C, F, T_f]
        auto mix_mag = stft_utils::magnitude(spec);

        std::vector<torch::Tensor> srcs;
        for (int c = 0; c < cfg.C; ++c) {
            auto m = masks[c];              // [F, T_f]
            auto est_mag = m * mix_mag.squeeze(0);
            auto est_spec = stft_utils::polar_to_complex(est_mag, phi.squeeze(0));
            auto wav = stft_utils::istft(est_spec.unsqueeze(0), stft_cfg, T);
            srcs.push_back(wav);             // [1, T]
        }
        separated = torch::cat(srcs, 0);    // [C, T]
        std::cout << "U-Net separation complete.\n";
    }

    // Save outputs
    for (int c = 0; c < cfg.C; ++c) {
        auto path = fs::path(cfg.output_dir) / ("source_" + std::to_string(c+1) + ".wav");
        audio::save_wav(path.string(), separated[c], cfg.sr);
        std::cout << "  → " << path << "\n";
    }

    // Evaluate if references provided
    if (!cfg.ref_s1.empty() && !cfg.ref_s2.empty()) {
        auto ref1 = audio::load_wav(cfg.ref_s1, cfg.sr).squeeze(0);
        auto ref2 = audio::load_wav(cfg.ref_s2, cfg.sr).squeeze(0);
        auto mix  = mix_wav.squeeze(0);

        int64_t minT = std::min({separated.size(1), ref1.size(0),
                                  ref2.size(0), mix.size(0)});
        auto sl = torch::indexing::Slice(0, minT);
        auto s1 = separated[0].index({sl});
        auto s2 = separated[1].index({sl});
        auto r1 = ref1.index({sl});
        auto r2 = ref2.index({sl});
        auto mx = mix.index({sl});

        // Try both permutations
        auto m1 = metrics::evaluate(s1, r1, mx, cfg.sr);
        auto m2 = metrics::evaluate(s2, r2, mx, cfg.sr);
        auto m1r = metrics::evaluate(s1, r2, mx, cfg.sr);
        auto m2r = metrics::evaluate(s2, r1, mx, cfg.sr);

        float snri_p1 = (m1.si_snri + m2.si_snri) / 2;
        float snri_p2 = (m1r.si_snri + m2r.si_snri) / 2;

        auto best = (snri_p1 >= snri_p2) ? std::make_pair(m1, m2)
                                          : std::make_pair(m1r, m2r);

        std::cout << "\n╔══ Evaluation Results ══╗\n"
                  << "  Source 1:  SI-SNRi=" << std::fixed << std::setprecision(2)
                  << best.first.si_snri << " dB  SDRi=" << best.first.sdri
                  << " dB  STOI=" << std::setprecision(3) << best.first.stoi << "\n"
                  << "  Source 2:  SI-SNRi=" << best.second.si_snri
                  << " dB  SDRi=" << best.second.sdri
                  << " dB  STOI=" << std::setprecision(3) << best.second.stoi << "\n"
                  << "  Average:   SI-SNRi=" << (best.first.si_snri+best.second.si_snri)/2
                  << " dB\n"
                  << "╚════════════════════════╝\n";
    }

    return 0;
}
