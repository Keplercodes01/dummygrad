#pragma once
// Data loading: mini-batches over in-memory tensors, and token streams for language models.
// Batches are built on the CPU and copied to the requested device.

#include "tensor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Batch {
    std::shared_ptr<Tensor> inputs;
    std::shared_ptr<Tensor> targets;  // null when the loader has no targets
};

// Mini-batches along the first dimension of `inputs` (and `targets`). Shuffling is a
// deterministic function of (seed, epoch), so a resumed run sees the same order.
//
//   DataLoader loader(X, Y, /*batch_size=*/64, /*shuffle=*/true, /*seed=*/0);
//   for (int64_t epoch = 0; epoch < 10; ++epoch) {
//       loader.set_epoch(epoch);
//       for (auto [x, y] : loader) { ... }
//   }
class DataLoader {
public:
    DataLoader(const std::shared_ptr<Tensor>& inputs, const std::shared_ptr<Tensor>& targets, int64_t batch_size,
               bool shuffle = true, uint64_t seed = 0, bool drop_last = false, Device device = Device::CPU)
        : batch_size_(batch_size), shuffle_(shuffle), seed_(seed), drop_last_(drop_last), device_(device) {
        if (!inputs || inputs->ndim() < 1) throw std::invalid_argument("DataLoader: inputs must have a sample dimension");
        if (batch_size <= 0) throw std::invalid_argument("DataLoader: batch_size must be positive");
        inputs_ = Host(inputs);
        if (targets) {
            if (targets->ndim() < 1 || targets->shape[0] != inputs->shape[0]) {
                throw std::invalid_argument("DataLoader: inputs and targets must have the same number of samples");
            }
            targets_ = Host(targets);
            has_targets_ = true;
        }
        order_.resize(inputs->shape[0]);
        set_epoch(0);
    }

    int64_t size() const { return (int64_t)order_.size(); }

    int64_t num_batches() const {
        int64_t n = size();
        return drop_last_ ? n / batch_size_ : (n + batch_size_ - 1) / batch_size_;
    }

    // Reorders the samples for this epoch (a no-op without shuffling).
    void set_epoch(int64_t epoch) {
        std::iota(order_.begin(), order_.end(), int64_t(0));
        if (shuffle_) {
            std::mt19937_64 rng(seed_ * 0x9E3779B97F4A7C15ULL + (uint64_t)epoch);
            std::shuffle(order_.begin(), order_.end(), rng);
        }
    }

    Batch batch(int64_t i) const {
        if (i < 0 || i >= num_batches()) throw std::out_of_range("DataLoader: batch index out of range");
        int64_t begin = i * batch_size_;
        int64_t end = std::min(begin + batch_size_, size());
        Batch b;
        b.inputs = inputs_.gather(order_, begin, end, device_);
        if (has_targets_) b.targets = targets_.gather(order_, begin, end, device_);
        return b;
    }

    class iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Batch;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = Batch;

        iterator(const DataLoader* loader, int64_t i) : loader_(loader), i_(i) {}
        Batch operator*() const { return loader_->batch(i_); }
        iterator& operator++() { ++i_; return *this; }
        bool operator!=(const iterator& o) const { return i_ != o.i_; }
        bool operator==(const iterator& o) const { return i_ == o.i_; }

    private:
        const DataLoader* loader_;
        int64_t i_;
    };
    iterator begin() const { return iterator(this, 0); }
    iterator end() const { return iterator(this, num_batches()); }

private:
    // Contiguous host copy of a tensor, viewed as rows of `row_bytes` along dimension 0.
    struct Host {
        std::shared_ptr<Tensor> data;
        std::vector<int64_t> row_shape;
        size_t row_bytes = 0;

        Host() = default;
        explicit Host(const std::shared_ptr<Tensor>& t) {
            data = make_contiguous(t->device == Device::CPU ? t : t->cpu());
            row_shape.assign(t->shape.begin() + 1, t->shape.end());
            int64_t row = 1;
            for (int64_t d : row_shape) row *= d;
            row_bytes = (size_t)row * dtype_size(t->dtype);
        }

        std::shared_ptr<Tensor> gather(const std::vector<int64_t>& order, int64_t begin, int64_t end, Device device) const {
            std::vector<int64_t> shape{end - begin};
            shape.insert(shape.end(), row_shape.begin(), row_shape.end());
            auto out = std::make_shared<Tensor>(shape, Device::CPU, data->dtype, false);
            const char* src = static_cast<const char*>(data->data_ptr<void>());
            char* dst = static_cast<char*>(out->data_ptr<void>());
            for (int64_t i = begin; i < end; ++i) {
                std::memcpy(dst + (i - begin) * row_bytes, src + order[i] * row_bytes, row_bytes);
            }
            if (device == Device::CPU) return out;
            auto moved = out->to(device);
            moved->requires_grad = false;
            return moved;
        }
    };

    Host inputs_, targets_;
    bool has_targets_ = false;
    std::vector<int64_t> order_;
    int64_t batch_size_;
    bool shuffle_;
    uint64_t seed_;
    bool drop_last_;
    Device device_;
};

// A stream of token ids for language modelling, sampled as windows where the targets are the
// inputs shifted by one token. Ids are stored in Float32 tensors (exact up to 2^24), following
// the engine's convention for integer data.
//
//   auto data = TokenDataset::from_file("train.bin");          // uint16 ids, as written by nanoGPT
//   auto [train, val] = data.split(0.9);
//   std::mt19937 rng(0);
//   Batch b = train.sample(/*batch_size=*/32, /*seq_len=*/256, rng, Device::CUDA);
class TokenDataset {
public:
    enum class Format { UInt16, Int32 };

    TokenDataset() = default;
    explicit TokenDataset(std::vector<int32_t> tokens) : tokens_(std::move(tokens)) {
        for (int32_t t : tokens_) {
            if (t < 0) throw std::invalid_argument("TokenDataset: token ids must be non-negative");
            vocab_size_ = std::max(vocab_size_, t + 1);
        }
    }

    // Raw little-endian token ids, e.g. the uint16 .bin files written by nanoGPT's prepare scripts.
    static TokenDataset from_file(const std::string& path, Format format = Format::UInt16) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("TokenDataset: cannot open " + path);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        size_t width = (format == Format::UInt16) ? 2 : 4;
        if (bytes.size() % width != 0) throw std::runtime_error("TokenDataset: " + path + " size is not a multiple of the token width");
        std::vector<int32_t> tokens(bytes.size() / width);
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (format == Format::UInt16) {
                uint16_t v;
                std::memcpy(&v, bytes.data() + i * 2, 2);
                tokens[i] = v;
            } else {
                std::memcpy(&tokens[i], bytes.data() + i * 4, 4);
            }
        }
        return TokenDataset(std::move(tokens));
    }

    // Every byte of the file is a token (vocabulary of 256): character-level models on raw text.
    static TokenDataset from_bytes(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("TokenDataset: cannot open " + path);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return from_string(std::string(bytes.begin(), bytes.end()));
    }

    static TokenDataset from_string(const std::string& text) {
        std::vector<int32_t> tokens(text.size());
        for (size_t i = 0; i < text.size(); ++i) tokens[i] = (unsigned char)text[i];
        TokenDataset d(std::move(tokens));
        d.vocab_size_ = 256;
        return d;
    }

    int64_t size() const { return (int64_t)tokens_.size(); }
    // Largest id + 1 (256 for byte-level data).
    int32_t vocab_size() const { return vocab_size_; }
    const std::vector<int32_t>& tokens() const { return tokens_; }

    // First `fraction` of the stream and the rest, e.g. split(0.9) for train/validation.
    std::pair<TokenDataset, TokenDataset> split(double fraction) const {
        if (fraction <= 0.0 || fraction >= 1.0) throw std::invalid_argument("TokenDataset::split: fraction must be in (0, 1)");
        auto cut = tokens_.begin() + (int64_t)(fraction * (double)tokens_.size());
        TokenDataset a(std::vector<int32_t>(tokens_.begin(), cut)), b(std::vector<int32_t>(cut, tokens_.end()));
        a.vocab_size_ = b.vocab_size_ = vocab_size_;
        return {std::move(a), std::move(b)};
    }

    // batch_size windows at random positions: inputs [B, T] and targets [B, T] (inputs shifted by one).
    Batch sample(int64_t batch_size, int64_t seq_len, std::mt19937& rng, Device device = Device::CPU) const {
        check_window(seq_len);
        std::uniform_int_distribution<int64_t> start(0, size() - seq_len - 1);
        std::vector<int64_t> starts(batch_size);
        for (auto& s : starts) s = start(rng);
        return make_batch(starts, seq_len, device);
    }

    // Number of non-overlapping windows of seq_len tokens (for evaluating on the whole stream).
    int64_t num_windows(int64_t seq_len) const {
        check_window(seq_len);
        return (size() - 1) / seq_len;
    }

    // Consecutive non-overlapping windows first_window, first_window + 1, ... (up to batch_size,
    // fewer at the end of the stream).
    Batch windows(int64_t first_window, int64_t batch_size, int64_t seq_len, Device device = Device::CPU) const {
        int64_t total = num_windows(seq_len);
        if (first_window < 0 || first_window >= total) throw std::out_of_range("TokenDataset::windows: window index out of range");
        std::vector<int64_t> starts;
        for (int64_t w = first_window; w < std::min(first_window + batch_size, total); ++w) starts.push_back(w * seq_len);
        return make_batch(starts, seq_len, device);
    }

private:
    void check_window(int64_t seq_len) const {
        if (seq_len <= 0 || size() <= seq_len) {
            throw std::invalid_argument("TokenDataset: need more than seq_len tokens (have " + std::to_string(size()) + ")");
        }
    }

    Batch make_batch(const std::vector<int64_t>& starts, int64_t seq_len, Device device) const {
        int64_t B = (int64_t)starts.size();
        auto x = std::make_shared<Tensor>(std::vector<int64_t>{B, seq_len}, Device::CPU, DType::Float32, false);
        auto y = std::make_shared<Tensor>(std::vector<int64_t>{B, seq_len}, Device::CPU, DType::Float32, false);
        float* xp = x->data_ptr<float>();
        float* yp = y->data_ptr<float>();
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < seq_len; ++t) {
                xp[b * seq_len + t] = (float)tokens_[starts[b] + t];
                yp[b * seq_len + t] = (float)tokens_[starts[b] + t + 1];
            }
        }
        Batch out{x, y};
        if (device != Device::CPU) {
            out.inputs = x->to(device);
            out.targets = y->to(device);
            out.inputs->requires_grad = out.targets->requires_grad = false;
        }
        return out;
    }

    std::vector<int32_t> tokens_;
    int32_t vocab_size_ = 0;
};
