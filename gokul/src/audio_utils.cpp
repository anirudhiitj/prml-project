// ─────────────────────────────────────────────────────────────────────────────
// audio_utils.cpp — WAV I/O implementation
// ─────────────────────────────────────────────────────────────────────────────
#include "audio_utils.h"
#include <sndfile.h>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <unistd.h>

namespace audio {

torch::Tensor load_wav(const std::string& path, int target_sr) {
    SF_INFO info{};
    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &info);
    if (!file)
        throw std::runtime_error("Cannot open WAV: " + path +
                                 " (" + sf_strerror(nullptr) + ")");

    if (info.samplerate != target_sr) {
        sf_close(file);
        throw std::runtime_error("Sample rate mismatch in " + path +
                                 ": expected " + std::to_string(target_sr) +
                                 ", got " + std::to_string(info.samplerate));
    }

    int64_t frames   = info.frames;
    int     channels = info.channels;
    std::vector<float> buf(frames * channels);
    sf_readf_float(file, buf.data(), frames);
    sf_close(file);

    auto t = torch::from_blob(buf.data(), {frames, channels}, torch::kFloat32).clone();
    if (channels > 1) t = t.mean(1);       // average to mono → [T]
    else               t = t.squeeze(1);    // [T]
    return t.unsqueeze(0);                  // [1, T]
}

void save_wav(const std::string& path, torch::Tensor w, int sample_rate) {
    w = w.contiguous().to(torch::kCPU, torch::kFloat32);
    if (w.dim() == 2) w = w.squeeze(0);     // [T]

    SF_INFO info{};
    info.samplerate = sample_rate;
    info.channels   = 1;
    info.format     = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* file = sf_open(path.c_str(), SFM_WRITE, &info);
    if (!file)
        throw std::runtime_error("Cannot create WAV: " + path);
    sf_writef_float(file, w.data_ptr<float>(), w.size(0));
    sf_close(file);
}

torch::Tensor fix_length(torch::Tensor w, int64_t length) {
    int orig_dim = w.dim();
    if (orig_dim == 1) w = w.unsqueeze(0);
    int64_t T = w.size(1);
    if (T >= length)
        w = w.index({torch::indexing::Slice(),
                     torch::indexing::Slice(0, length)});
    else
        w = torch::cat({w, torch::zeros({w.size(0), length - T}, w.options())}, 1);
    if (orig_dim == 1) w = w.squeeze(0);
    return w;
}

torch::Tensor load_wav_resample(const std::string& path, int target_sr) {
    // Try probing the file with libsndfile
    SF_INFO info{};
    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &info);

    if (file) {
        int actual_sr = info.samplerate;
        int channels  = info.channels;
        sf_close(file);

        if (actual_sr == target_sr && channels == 1)
            return load_wav(path, target_sr);

        // Need conversion
        if (actual_sr != target_sr)
            std::cout << "[Audio] Resampling " << actual_sr
                      << " -> " << target_sr << " Hz\n";
        if (channels > 1)
            std::cout << "[Audio] Mixing " << channels
                      << " channels -> mono\n";
    } else {
        // libsndfile can't open it (MP3/MP4/etc.) — try ffmpeg
        std::cout << "[Audio] Converting via ffmpeg\n";
    }

    std::string tmp = "/tmp/prml_sep_" + std::to_string(getpid()) + ".wav";
    std::string cmd = "ffmpeg -y -loglevel error -i \"" + path +
                      "\" -ar " + std::to_string(target_sr) +
                      " -ac 1 \"" + tmp + "\"";
    int ret = std::system(cmd.c_str());
    if (ret != 0)
        throw std::runtime_error(
            "Cannot load audio: " + path +
            ". Install ffmpeg or convert to " +
            std::to_string(target_sr) + " Hz mono WAV.");
    auto wav = load_wav(tmp, target_sr);
    std::remove(tmp.c_str());
    return wav;
}

}  // namespace audio
