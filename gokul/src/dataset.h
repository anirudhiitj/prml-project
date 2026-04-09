#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// dataset.h — LibriMix dataset for 2-speaker separation
// ─────────────────────────────────────────────────────────────────────────────
#include <torch/torch.h>
#include <string>
#include <vector>

namespace data {

/// LibriMix CSV dataset.
/// Returns Example<Tensor, Tensor>: data=[T], target=[C,T]
class LibriMixDataset : public torch::data::datasets::Dataset<LibriMixDataset> {
public:
    LibriMixDataset(const std::string& csv_path,
                    int64_t segment_len = 32000,
                    int sample_rate     = 8000,
                    bool augment        = false);

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    struct Entry { std::string mix, s1, s2; };
    std::vector<Entry> entries_;
    int64_t seg_len_;
    int     sr_;
    bool    aug_;
};

}  // namespace data
