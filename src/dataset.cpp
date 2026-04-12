// ─────────────────────────────────────────────────────────────────────────────
// dataset.cpp — LibriMix dataset implementation
//
// Robust CSV parser that handles multiple LibriMix CSV formats:
//   Format A: mixture_ID, mixture_path, source_1_path, source_2_path
//   Format B: mixture_ID, mixture_path, source_1_path, source_2_path, noise_path
//   Format C: mixture_ID, mixture_path, source_1_path, source_2_path, length
// Automatically detects columns by header names.
// ─────────────────────────────────────────────────────────────────────────────
#include "dataset.h"
#include "audio_utils.h"
#include "augmentation.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

namespace data {

static thread_local std::mt19937 rng{std::random_device{}()};

// Parse CSV header to find column indices
struct CSVColumns {
    int mix_col = -1;
    int s1_col  = -1;
    int s2_col  = -1;
};

static CSVColumns detect_columns(const std::string& header) {
    CSVColumns cols;
    std::istringstream ss(header);
    std::string tok;
    int idx = 0;
    while (std::getline(ss, tok, ',')) {
        // Trim whitespace
        tok.erase(0, tok.find_first_not_of(" \t\r\n"));
        tok.erase(tok.find_last_not_of(" \t\r\n") + 1);
        // Convert to lowercase for matching
        std::string lower;
        for (char c : tok) lower += std::tolower(c);

        if (lower.find("mixture_path") != std::string::npos ||
            lower.find("mix_path") != std::string::npos) {
            cols.mix_col = idx;
        } else if (lower.find("source_1") != std::string::npos ||
                   lower.find("s1") != std::string::npos) {
            cols.s1_col = idx;
        } else if (lower.find("source_2") != std::string::npos ||
                   lower.find("s2") != std::string::npos) {
            cols.s2_col = idx;
        }
        idx++;
    }

    // Fallback: if header detection failed, use standard positions
    if (cols.mix_col < 0 || cols.s1_col < 0 || cols.s2_col < 0) {
        cols.mix_col = 1;  // index 1 = mixture_path
        cols.s1_col  = 2;  // index 2 = source_1_path
        cols.s2_col  = 3;  // index 3 = source_2_path
    }

    return cols;
}

LibriMixDataset::LibriMixDataset(const std::string& csv_path,
                                 int64_t segment_len, int sample_rate,
                                 bool augment)
    : seg_len_(segment_len), sr_(sample_rate), aug_(augment) {

    std::ifstream f(csv_path);
    if (!f.is_open()) throw std::runtime_error("Cannot open CSV: " + csv_path);

    std::string header;
    std::getline(f, header);  // read header

    auto cols = detect_columns(header);
    int max_col = std::max({cols.mix_col, cols.s1_col, cols.s2_col});

    std::string line;
    int line_num = 1;
    int skipped = 0;
    while (std::getline(f, line)) {
        line_num++;
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> row;
        while (std::getline(ss, tok, ',')) {
            tok.erase(0, tok.find_first_not_of(" \t\r\n"));
            tok.erase(tok.find_last_not_of(" \t\r\n") + 1);
            row.push_back(tok);
        }

        if ((int)row.size() <= max_col) {
            skipped++;
            continue;
        }

        std::string mix_path = row[cols.mix_col];
        std::string s1_path  = row[cols.s1_col];
        std::string s2_path  = row[cols.s2_col];

        // Verify files exist (only check first few to avoid slow startup)
        if (entries_.size() < 5) {
            if (!fs::exists(mix_path)) {
                std::cerr << "[LibriMix] WARNING: File not found: " << mix_path
                          << " (line " << line_num << ")\n";
                skipped++;
                continue;
            }
        }

        entries_.push_back({mix_path, s1_path, s2_path});
    }

    std::cout << "[LibriMix] " << entries_.size() << " examples from "
              << csv_path;
    if (skipped > 0)
        std::cout << " (" << skipped << " skipped)";
    std::cout << "\n";
    std::cout << "[LibriMix] CSV columns: mix=" << cols.mix_col
              << " s1=" << cols.s1_col << " s2=" << cols.s2_col << "\n";

    if (entries_.empty()) {
        throw std::runtime_error("No valid entries found in " + csv_path +
                                 ". Check that file paths in the CSV are correct.");
    }
}

torch::data::Example<> LibriMixDataset::get(size_t idx) {
    auto& e = entries_.at(idx);

    // Load audio with error handling
    torch::Tensor mix, src1, src2;
    try {
        mix  = audio::load_wav(e.mix, sr_);
        src1 = audio::load_wav(e.s1,  sr_);
        src2 = audio::load_wav(e.s2,  sr_);
    } catch (const std::exception& ex) {
        // On load failure, return zeros so training doesn't crash
        std::cerr << "[LibriMix] WARNING: Failed to load index " << idx
                  << ": " << ex.what() << "\n";
        int64_t len = seg_len_ > 0 ? seg_len_ : 32000;
        return {torch::zeros({len}), torch::zeros({2, len})};
    }

    auto sources = torch::cat({src1, src2}, 0);  // [2, T]

    // Random crop or pad to segment length
    if (seg_len_ > 0) {
        int64_t T = mix.size(1);
        if (T > seg_len_) {
            int64_t s = std::uniform_int_distribution<int64_t>(0, T-seg_len_)(rng);
            auto sl = torch::indexing::Slice(s, s + seg_len_);
            mix     = mix.index({torch::indexing::Slice(), sl});
            sources = sources.index({torch::indexing::Slice(), sl});
        } else if (T < seg_len_) {
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
