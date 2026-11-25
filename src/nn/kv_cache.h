#pragma once
// Keys and values of the tokens an attention layer has already seen, so that generating a token costs one
// query against them instead of a whole forward pass. For inference only.

#include "functional.h"

#include <utility>
#include <vector>

struct KVCache {
    std::shared_ptr<Tensor> k, v;  // [batch, kv heads, capacity, head dim]
    int64_t length = 0;            // tokens stored
    int64_t reserve = 0;           // tokens to make room for at first; storage doubles when it runs out

    // Appends keys and values [batch, kv heads, S, head dim] and returns views of all of them
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> append(const std::shared_ptr<Tensor>& k_new, const std::shared_ptr<Tensor>& v_new) {
        if (grad_enabled()) throw std::runtime_error("KVCache: gradients do not flow through a cache; use it inside a NoGradGuard");
        if (k_new->ndim() != 4 || k_new->shape != v_new->shape) throw std::invalid_argument("KVCache: keys and values must both be [batch, heads, time, head dim]");
        auto kd = make_contiguous(k_new), vd = make_contiguous(v_new);
        int64_t S = kd->shape[2];
        if (length > 0) {
            bool fits = k->shape[0] == kd->shape[0] && k->shape[1] == kd->shape[1] && k->shape[3] == kd->shape[3] &&
                        k->dtype == kd->dtype && k->device == kd->device;
            if (!fits) throw std::invalid_argument("KVCache: the batch, heads, dtype or device changed; start a new cache (reset())");
        }
        bool reusable = k && k->shape[0] == kd->shape[0] && k->shape[1] == kd->shape[1] && k->shape[3] == kd->shape[3] &&
                        k->dtype == kd->dtype && k->device == kd->device;
        int64_t capacity = reusable ? k->shape[2] : 0;
        if (length + S > capacity) grow(*kd, std::max({length + S, reserve, 2 * capacity}));
        copy_tokens(*k, length, *kd, S);
        copy_tokens(*v, length, *vd, S);
        length += S;
        return {narrow(k, 2, 0, length), narrow(v, 2, 0, length)};
    }

    // Forgets every token, keeping the storage
    void reset() { length = 0; }
    // Forgets the tokens past the first n
    void truncate(int64_t n) {
        if (n < 0 || n > length) throw std::invalid_argument("KVCache: cannot keep more tokens than are stored");
        length = n;
    }
    // Rearranges the batch: row i becomes the old row rows[i] (beam search; a row may repeat)
    void select_rows(const std::vector<int64_t>& rows) {
        if (!k) return;
        k = take_rows(*k, rows);
        v = take_rows(*v, rows);
    }

private:
    // dst and src are dense [B, H, capacity, D]: src's first count tokens go to dst's tokens from dst_offset on
    static void copy_tokens(Tensor& dst, int64_t dst_offset, const Tensor& src, int64_t count) {
        if (count == 0) return;
        size_t elem = dtype_size(dst.dtype);
        int64_t rows = dst.shape[0] * dst.shape[1], D = dst.shape[3];
        size_t width = (size_t)(count * D) * elem, dst_pitch = (size_t)(dst.shape[2] * D) * elem, src_pitch = (size_t)(src.shape[2] * D) * elem;
        char* d = dst.data_ptr<char>() + (size_t)(dst_offset * D) * elem;
        const char* s = src.data_ptr<char>();
#ifdef USE_CUDA
        if (dst.device == Device::CUDA) {
            check_cuda(cudaMemcpy2DAsync(d, dst_pitch, s, src_pitch, width, (size_t)rows, cudaMemcpyDeviceToDevice), "KVCache");
            dst.storage->bump_version();
            return;
        }
#endif
        for (int64_t r = 0; r < rows; ++r) std::memcpy(d + (size_t)r * dst_pitch, s + (size_t)r * src_pitch, width);
        dst.storage->bump_version();
    }

    void grow(const Tensor& like, int64_t capacity) {
        std::vector<int64_t> shape{like.shape[0], like.shape[1], capacity, like.shape[3]};
        auto nk = std::make_shared<Tensor>(shape, like.device, like.dtype, false), nv = std::make_shared<Tensor>(shape, like.device, like.dtype, false);
        if (length > 0) {
            copy_tokens(*nk, 0, *k, length);
            copy_tokens(*nv, 0, *v, length);
        }
        k = nk;
        v = nv;
    }

    static std::shared_ptr<Tensor> take_rows(const Tensor& t, const std::vector<int64_t>& rows) {
        auto shape = t.shape;
        shape[0] = (int64_t)rows.size();
        auto out = std::make_shared<Tensor>(shape, t.device, t.dtype, false);
        size_t bytes = (size_t)(t.shape[1] * t.shape[2] * t.shape[3]) * dtype_size(t.dtype);
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] < 0 || rows[i] >= t.shape[0]) throw std::out_of_range("KVCache: row out of range");
            const char* s = t.data_ptr<char>() + (size_t)rows[i] * bytes;
            char* d = out->data_ptr<char>() + i * bytes;
#ifdef USE_CUDA
            if (t.device == Device::CUDA) {
                check_cuda(cudaMemcpyAsync(d, s, bytes, cudaMemcpyDeviceToDevice), "KVCache");
                continue;
            }
#endif
            std::memcpy(d, s, bytes);
        }
        return out;
    }
};

// Scores of S new queries against past + S keys: a query sees every past token and the new ones up to itself
inline std::shared_ptr<Tensor> cached_causal_mask(int64_t S, int64_t past, Device device) {
    std::vector<float> m((size_t)(S * (past + S)), 0.0f);
    for (int64_t i = 0; i < S; ++i)
        for (int64_t j = past + i + 1; j < past + S; ++j) m[(size_t)(i * (past + S) + j)] = -1e9f;
    return op_detail::from_values(m, {S, past + S}, device);
}
