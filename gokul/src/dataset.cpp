// ─────────────────────────────────────────────────────────────────────────────
// dataset.cpp — LibriMix dataset implementation
// ─────────────────────────────────────────────────────────────────────────────
#include "dataset.h"
#include "audio_utils.h"
#include "augmentation.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <random>

namespace data {

static thread_local std::mt19937 rng{std::random_device{}()};

LibriMixDataset::LibriMixDataset(const std::string& csv_path,
                                 int64_t segment_len, int sample_rate,
                                 bool augment)
    : seg_len_(segment_len), sr_(sample_rate), aug_(augment) {

    std::ifstream f(csv_path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + csv_path);

    std::string line;
    std::getline(f, line);  // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ',')) {
            tok.erase(0, tok.find_first_not_of(" \t\r\n"));
            tok.erase(tok.find_last_not_of(" \t\r\n") + 1);
            cols.push_back(tok);
        }
        if (cols.size() >= 4)
            entries_.push_back({cols[1], cols[2], cols[3]});
    }
    std::cout << "[LibriMix] " << entries_.size() << " examples from "
              << csv_path << "\n";
}

torch::data::Example<> LibriMixDataset::get(size_t idx) {
    auto& e = entries_.at(idx);
    auto mix  = audio::load_wav(e.mix, sr_);
    auto src1 = audio::load_wav(e.s1,  sr_);
    auto src2 = audio::load_wav(e.s2,  sr_);
    auto sources = torch::cat({src1, src2}, 0);  // [2, T]

    // Random crop or pad to segment length
    if (seg_len_ > 0) {
        int64_t T = mix.size(1);
        if (T > seg_len_) {
            int64_t s = std::uniform_int_distribution<int64_t>(0, T-seg_len_)(rng);
            auto sl = torch::indexing::Slice(s, s + seg_len_);
            mix     = mix.index({torch::indexing::Slice(), sl});
            sources = sources.index({torch::indexing::Slice(), sl});
        } else {
            mix     = audio::fix_length(mix, seg_len_);
            sources = torch::cat({
                audio::fix_length(sources[0].unsqueeze(0), seg_len_),
                audio::fix_length(sources[1].unsqueeze(0), seg_len_)
            }, 0);
        }
    }

    if (aug_) {
        auto [am, as] = augment::apply(mix, sources);
        mix = am; sources = as;
    }

    return {mix.squeeze(0), sources};  // [T], [C, T]
}

torch::optional<size_t> LibriMixDataset::size() const {
    return entries_.size();
}

}  // namespace data
