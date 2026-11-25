#pragma once
// Data loading: mini-batches of tensors or of a Dataset's samples, and token streams for language models. Batches
// are built on the CPU, on background threads when asked to, and copied to the requested device.

#include "mapped_file.h"
#include "prefetch.h"
#include "tensor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
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

namespace data_detail {

inline Batch to_device(Batch b, Device device) {
    if (device == Device::CPU) return b;
    for (auto* t : {&b.inputs, &b.targets}) {
        if (!*t) continue;
        *t = (*t)->to(device);
        (*t)->requires_grad = false;
    }
    return b;
}

// Contiguous host copy of a tensor, viewed as rows along dimension 0
struct Rows {
    std::shared_ptr<Tensor> data;
    std::vector<int64_t> row_shape;
    size_t row_bytes = 0;

    Rows() = default;
    explicit Rows(const std::shared_ptr<Tensor>& t) {
        data = make_contiguous(t->device == Device::CPU ? t : t->cpu());
        row_shape.assign(t->shape.begin() + 1, t->shape.end());
        int64_t row = 1;
        for (int64_t d : row_shape) row *= d;
        row_bytes = (size_t)row * dtype_size(t->dtype);
    }

    std::shared_ptr<Tensor> gather(const int64_t* rows, int64_t count) const {
        std::vector<int64_t> shape{count};
        shape.insert(shape.end(), row_shape.begin(), row_shape.end());
        auto out = std::make_shared<Tensor>(shape, Device::CPU, data->dtype, false);
        const char* src = static_cast<const char*>(data->data_ptr<void>());
        char* dst = static_cast<char*>(out->data_ptr<void>());
        for (int64_t i = 0; i < count; ++i) std::memcpy(dst + (size_t)i * row_bytes, src + (size_t)rows[i] * row_bytes, row_bytes);
        return out;
    }
};

// Samples of one shape and dtype as a batch, with a new first dimension
inline std::shared_ptr<Tensor> stack_samples(const std::vector<std::shared_ptr<Tensor>>& samples) {
    const auto& first = samples[0];
    std::vector<int64_t> shape{(int64_t)samples.size()};
    shape.insert(shape.end(), first->shape.begin(), first->shape.end());
    auto out = std::make_shared<Tensor>(shape, Device::CPU, first->dtype, false);
    size_t bytes = (size_t)first->size() * dtype_size(first->dtype);
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!samples[i] || samples[i]->shape != first->shape || samples[i]->dtype != first->dtype) {
            throw std::invalid_argument("DataLoader: the samples of a batch must have one shape and dtype");
        }
        auto host = make_contiguous(samples[i]->device == Device::CPU ? samples[i] : samples[i]->cpu());
        if (bytes > 0) std::memcpy(out->data_ptr<char>() + i * bytes, host->data_ptr<char>(), bytes);
    }
    return out;
}

}  // namespace data_detail

// Samples made on demand: files to read, images to augment. get(i) returns one sample without a batch dimension.
// With loader workers it is called from several threads at once.
class Dataset {
public:
    virtual ~Dataset() = default;
    virtual int64_t size() const = 0;
    virtual Batch get(int64_t i) const = 0;
};

// Mini-batches along the first dimension of `inputs` (and `targets`), or of a Dataset. Shuffling is a
// deterministic function of (seed, epoch), so a resumed run sees the same order.
//
//   DataLoader loader(X, Y, /*batch_size=*/64, /*shuffle=*/true, /*seed=*/0);
//   loader.prefetch(/*workers=*/2);              // optional: batches are made while the model trains
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
        if (targets && (targets->ndim() < 1 || targets->shape[0] != inputs->shape[0])) {
            throw std::invalid_argument("DataLoader: inputs and targets must have the same number of samples");
        }
        data_detail::Rows x(inputs), y;
        bool has_targets = targets != nullptr;
        if (has_targets) y = data_detail::Rows(targets);
        make_ = [x, y, has_targets](const int64_t* rows, int64_t count) {
            return Batch{x.gather(rows, count), has_targets ? y.gather(rows, count) : nullptr};
        };
        start(inputs->shape[0]);
    }

    DataLoader(std::shared_ptr<const Dataset> dataset, int64_t batch_size, bool shuffle = true, uint64_t seed = 0,
               bool drop_last = false, Device device = Device::CPU)
        : batch_size_(batch_size), shuffle_(shuffle), seed_(seed), drop_last_(drop_last), device_(device) {
        if (!dataset) throw std::invalid_argument("DataLoader: null dataset");
        make_ = [dataset](const int64_t* rows, int64_t count) {
            std::vector<std::shared_ptr<Tensor>> x, y;
            for (int64_t i = 0; i < count; ++i) {
                Batch s = dataset->get(rows[i]);
                if (!s.inputs || (i > 0 && (s.targets != nullptr) != (y.size() == (size_t)i))) {
                    throw std::invalid_argument("DataLoader: every sample needs inputs, and all or none of them targets");
                }
                x.push_back(s.inputs);
                if (s.targets) y.push_back(s.targets);
            }
            return Batch{data_detail::stack_samples(x), y.empty() ? nullptr : data_detail::stack_samples(y)};
        };
        start(dataset->size());
    }

    // Makes batches on `workers` background threads, up to `ahead` of them before they are needed (0 workers: on the
    // calling thread, as they are needed). Applies to loops that start after the call.
    void prefetch(int workers, int ahead = 4) {
        if (workers < 0 || ahead < 1) throw std::invalid_argument("DataLoader::prefetch: bad arguments");
        workers_ = workers;
        ahead_ = ahead;
    }

    int64_t size() const { return (int64_t)order_->size(); }

    int64_t num_batches() const {
        int64_t n = size();
        return drop_last_ ? n / batch_size_ : (n + batch_size_ - 1) / batch_size_;
    }

    // Reorders the samples for this epoch (a no-op without shuffling).
    void set_epoch(int64_t epoch) {
        auto order = std::make_shared<std::vector<int64_t>>(order_->size());
        std::iota(order->begin(), order->end(), int64_t(0));
        if (shuffle_) {
            std::mt19937_64 rng(seed_ * 0x9E3779B97F4A7C15ULL + (uint64_t)epoch);
            std::shuffle(order->begin(), order->end(), rng);
        }
        order_ = order;
    }

    Batch batch(int64_t i) const {
        if (i < 0 || i >= num_batches()) throw std::out_of_range("DataLoader: batch index out of range");
        return data_detail::to_device(host_batch(*order_, i), device_);
    }

    class iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Batch;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = Batch;

        iterator(const DataLoader* loader, int64_t i, std::shared_ptr<Prefetcher<Batch>> ahead = nullptr)
            : loader_(loader), i_(i), ahead_(std::move(ahead)) {}
        Batch operator*() const {
            if (!ahead_) return loader_->batch(i_);
            while (taken_ <= i_) {
                ++taken_;
                ahead_->next(current_);
                if (taken_ == i_ + 1) current_ = data_detail::to_device(std::move(current_), loader_->device_);
            }
            return current_;
        }
        iterator& operator++() { ++i_; return *this; }
        bool operator!=(const iterator& o) const { return i_ != o.i_; }
        bool operator==(const iterator& o) const { return i_ == o.i_; }

    private:
        const DataLoader* loader_;
        int64_t i_;
        std::shared_ptr<Prefetcher<Batch>> ahead_;
        mutable int64_t taken_ = 0;
        mutable Batch current_;
    };
    iterator begin() const {
        if (workers_ == 0) return iterator(this, 0);
        auto order = order_;  // of the epoch this loop started in
        return iterator(this, 0, std::make_shared<Prefetcher<Batch>>([this, order](int64_t i) { return host_batch(*order, i); }, num_batches(), workers_, ahead_));
    }
    iterator end() const { return iterator(this, num_batches()); }

private:
    void start(int64_t samples) {
        if (batch_size_ <= 0) throw std::invalid_argument("DataLoader: batch_size must be positive");
        order_ = std::make_shared<std::vector<int64_t>>((size_t)samples);
        set_epoch(0);
    }

    Batch host_batch(const std::vector<int64_t>& order, int64_t i) const {
        int64_t begin = i * batch_size_;
        int64_t end = std::min(begin + batch_size_, (int64_t)order.size());
        return make_(order.data() + begin, end - begin);
    }

    std::function<Batch(const int64_t* rows, int64_t count)> make_;  // a host batch of these samples
    std::shared_ptr<const std::vector<int64_t>> order_;
    int64_t batch_size_;
    bool shuffle_;
    uint64_t seed_;
    bool drop_last_;
    Device device_;
    int workers_ = 0, ahead_ = 4;
};

// A stream of token ids for language modelling, sampled as windows where the targets are the
// inputs shifted by one token. Ids are stored in Float32 tensors (exact up to 2^24), following
// the engine's convention for integer data. Files are memory-mapped: only the windows used are read.
//
//   auto data = TokenDataset::from_file("train.bin");          // uint16 ids, as written by nanoGPT
//   auto [train, val] = data.split(0.9);
//   std::mt19937 rng(0);
//   Batch b = train.sample(/*batch_size=*/32, /*seq_len=*/256, rng, Device::CUDA);
class TokenDataset {
public:
    enum class Format { UInt16, Int32 };

    TokenDataset() = default;
    explicit TokenDataset(std::vector<int32_t> tokens) {
        for (int32_t t : tokens) if (t < 0) throw std::invalid_argument("TokenDataset: token ids must be non-negative");
        auto owned = std::make_shared<std::vector<int32_t>>(std::move(tokens));
        source_ = std::make_shared<Source>();
        source_->bytes = reinterpret_cast<const uint8_t*>(owned->data());
        source_->width = 4;
        source_->count = (int64_t)owned->size();
        source_->keep = owned;
        size_ = source_->count;
    }

    // Raw little-endian token ids, e.g. the uint16 .bin files written by nanoGPT's prepare scripts.
    static TokenDataset from_file(const std::string& path, Format format = Format::UInt16) {
        auto file = std::make_shared<MappedFile>(path);
        int width = (format == Format::UInt16) ? 2 : 4;
        if (file->size() % (size_t)width != 0) throw std::runtime_error("TokenDataset: " + path + " size is not a multiple of the token width");
        return over(file, width, 0);
    }

    // Every byte of the file is a token (vocabulary of 256): character-level models on raw text.
    static TokenDataset from_bytes(const std::string& path) { return over(std::make_shared<MappedFile>(path), 1, 256); }

    static TokenDataset from_string(const std::string& text) {
        auto owned = std::make_shared<std::string>(text);
        TokenDataset d;
        d.source_ = std::make_shared<Source>();
        d.source_->bytes = reinterpret_cast<const uint8_t*>(owned->data());
        d.source_->width = 1;
        d.source_->count = (int64_t)owned->size();
        d.source_->vocab_size = 256;
        d.source_->keep = owned;
        d.size_ = d.source_->count;
        return d;
    }

    int64_t size() const { return size_; }
    int32_t token(int64_t i) const {
        if (i < 0 || i >= size_) throw std::out_of_range("TokenDataset: token index out of range");
        return source_->at(begin_ + i);
    }
    // Largest id + 1 of the whole stream (256 for byte-level data); reads all of it the first time
    int32_t vocab_size() const {
        if (!source_) return 0;
        std::lock_guard<std::mutex> lock(source_->mutex);
        if (source_->vocab_size == 0) {
            for (int64_t i = 0; i < source_->count; ++i) source_->vocab_size = std::max(source_->vocab_size, source_->at(i) + 1);
        }
        return source_->vocab_size;
    }
    // A copy of the ids
    std::vector<int32_t> tokens() const {
        std::vector<int32_t> out((size_t)size_);
        for (int64_t i = 0; i < size_; ++i) out[(size_t)i] = source_->at(begin_ + i);
        return out;
    }

    // First `fraction` of the stream and the rest, e.g. split(0.9) for train/validation. Both share its memory.
    std::pair<TokenDataset, TokenDataset> split(double fraction) const {
        if (fraction <= 0.0 || fraction >= 1.0) throw std::invalid_argument("TokenDataset::split: fraction must be in (0, 1)");
        TokenDataset a = *this, b = *this;
        a.size_ = (int64_t)(fraction * (double)size_);
        b.begin_ = begin_ + a.size_;
        b.size_ = size_ - a.size_;
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
    struct Source {
        const uint8_t* bytes = nullptr;
        int width = 4;  // bytes per token
        int64_t count = 0;
        int32_t vocab_size = 0;  // 0: not counted yet
        std::mutex mutex;        // guards vocab_size
        std::shared_ptr<const void> keep;  // what owns the bytes

        int32_t at(int64_t i) const {
            if (width == 1) return bytes[i];
            if (width == 2) {
                uint16_t v;
                std::memcpy(&v, bytes + i * 2, 2);
                return v;
            }
            int32_t v;
            std::memcpy(&v, bytes + i * 4, 4);
            return v;
        }
    };

    static TokenDataset over(const std::shared_ptr<MappedFile>& file, int width, int32_t vocab_size) {
        TokenDataset d;
        d.source_ = std::make_shared<Source>();
        d.source_->bytes = file->data();
        d.source_->width = width;
        d.source_->count = (int64_t)(file->size() / (size_t)width);
        d.source_->vocab_size = vocab_size;
        d.source_->keep = file;
        d.size_ = d.source_->count;
        return d;
    }

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
                int64_t at = begin_ + starts[(size_t)b] + t;
                int32_t id = source_->at(at), next = source_->at(at + 1);
                if (id < 0 || next < 0) throw std::runtime_error("TokenDataset: negative token id in the stream");
                xp[b * seq_len + t] = (float)id;
                yp[b * seq_len + t] = (float)next;
            }
        }
        return data_detail::to_device(Batch{x, y}, device);
    }

    std::shared_ptr<Source> source_;
    int64_t begin_ = 0, size_ = 0;
};

// Batches of random windows, made on background threads without end. Batch i depends only on (seed, i), not on the
// number of workers, so a run resumed at step n with first = n sees what it would have seen.
//
//   TokenLoader loader(train, 32, 256, /*seed=*/0, Device::CUDA);
//   for (int64_t step = 0; step < steps; ++step) {
//       Batch b = loader.next();
//       ...
//   }
class TokenLoader {
public:
    TokenLoader(const TokenDataset& data, int64_t batch_size, int64_t seq_len, uint64_t seed = 0, Device device = Device::CPU,
                int workers = 1, int ahead = 4, int64_t first = 0)
        : device_(device),
          batches_([data, batch_size, seq_len, seed, first](int64_t i) {
              std::seed_seq mix{(uint32_t)seed, (uint32_t)(seed >> 32), (uint32_t)(first + i), (uint32_t)((uint64_t)(first + i) >> 32)};
              std::mt19937 rng(mix);
              return data.sample(batch_size, seq_len, rng, Device::CPU);
          }, -1, workers, ahead) {}

    Batch next() {
        Batch b;
        batches_.next(b);
        return data_detail::to_device(std::move(b), device_);
    }

private:
    Device device_;
    Prefetcher<Batch> batches_;
};
