#pragma once
#include "tensor.h"
#include "functional.h"
#include "node.h"
#include "ops.h"
#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <string>

#ifdef USE_CUDA
#include "cuda/cuda_ops.cuh"
#endif
#include "metal/metal_backend.h"

// ============================================================================
// Modern Masking Suite for Frontier LLM Architectures
// Supports:
// 1. Sliding Window Attention Mask (Mistral 7B, Gemma 2, Qwen 2)
// 2. Prefix-LM / Causal-with-Prefix Mask (T5, PaLM, Chameleon, FIM)
// 3. ALiBi Slope Bias Matrix (BLOOM, MPT, Falcon)
// 4. Arbitrary Key-Padding Mask (Variable-length Batched Attention)
// 5. Document-Packing Block-Diagonal Mask (LLaMA 3, DeepSeek Sample Multiplexing)
// ============================================================================

namespace masking {

// ----------------------------------------------------------------------------
// 1. Sliding Window Attention Mask Backward Autograd Node
// ----------------------------------------------------------------------------
struct SlidingWindowMaskBackward : public Node {
    int r, c, batch_size, window_size;

    SlidingWindowMaskBackward(int r, int c, int batch_size, int window_size)
        : r(r), c(c), batch_size(batch_size), window_size(window_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        auto self_grad = grads[0];
        auto ga = std::make_shared<Tensor>(self_grad->shape, self_grad->device, self_grad->dtype, false);
        ga->fill_(0.0f);

#ifdef USE_CUDA
        if (self_grad->device == Device::CUDA) {
            cuda::sliding_window_mask_backward(self_grad->data_ptr<float>(), ga->data_ptr<float>(),
                                               batch_size, r, window_size);
            return {ga};
        }
#endif
        if (self_grad->device == Device::MPS) {
            MetalBackend::get().sliding_window_mask_backward(self_grad->data_ptr<float>(), ga->data_ptr<float>(),
                                                             batch_size, r, window_size);
            return {ga};
        }

        const float* sg_ptr = self_grad->data_ptr<float>();
        float* ga_ptr = ga->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int idx = batch * r * c + i * c + j;
                    bool valid = (j <= i) && (i - j <= window_size);
                    ga_ptr[idx] = valid ? sg_ptr[idx] : 0.0f;
                }
            }
        }
        return {ga};
    }
};

// Applies Sliding Window Attention mask to scores tensor [..., seq_len, seq_len]
// Tokens attend only to the past window_size tokens: i - window_size <= j <= i
inline std::shared_ptr<Tensor> sliding_window_mask(const std::shared_ptr<Tensor>& scores, int window_size) {
    int ndim = scores->shape.size();
    if (ndim < 2) {
        throw std::invalid_argument("sliding_window_mask requires tensor of at least 2 dimensions");
    }
    int r = scores->shape[ndim - 2];
    int c = scores->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= scores->shape[i]; }

    bool req_grad = scores->requires_grad;
    auto out = std::make_shared<Tensor>(scores->shape, scores->device, scores->dtype, req_grad);

#ifdef USE_CUDA
    if (scores->device == Device::CUDA) {
        cuda::sliding_window_mask(scores->data_ptr<float>(), out->data_ptr<float>(), batch_size, r, window_size);
        if (req_grad) {
            auto grad_fn = std::make_shared<SlidingWindowMaskBackward>(r, c, batch_size, window_size);
            grad_fn->add_next_edge(get_grad_edge(scores).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif
    if (scores->device == Device::MPS) {
        MetalBackend::get().sliding_window_mask(scores->data_ptr<float>(), out->data_ptr<float>(),
                                                batch_size, r, window_size);
        if (req_grad) {
            auto grad_fn = std::make_shared<SlidingWindowMaskBackward>(r, c, batch_size, window_size);
            grad_fn->add_next_edge(get_grad_edge(scores).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }

    const float* s_ptr = scores->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int idx = batch * r * c + i * c + j;
                bool valid = (j <= i) && (i - j <= window_size);
                out_ptr[idx] = valid ? s_ptr[idx] : -1e9f;
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<SlidingWindowMaskBackward>(r, c, batch_size, window_size);
        grad_fn->add_next_edge(get_grad_edge(scores).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ----------------------------------------------------------------------------
// 2. Prefix-LM / Causal-with-Prefix Mask Backward Autograd Node
// ----------------------------------------------------------------------------
struct PrefixCausalMaskBackward : public Node {
    int r, c, batch_size, prefix_len;

    PrefixCausalMaskBackward(int r, int c, int batch_size, int prefix_len)
        : r(r), c(c), batch_size(batch_size), prefix_len(prefix_len) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        auto self_grad = grads[0];
        auto ga = std::make_shared<Tensor>(self_grad->shape, self_grad->device, self_grad->dtype, false);
        ga->fill_(0.0f);

#ifdef USE_CUDA
        if (self_grad->device == Device::CUDA) {
            cuda::prefix_causal_mask_backward(self_grad->data_ptr<float>(), ga->data_ptr<float>(),
                                              batch_size, r, prefix_len);
            return {ga};
        }
#endif

        const float* sg_ptr = self_grad->data_ptr<float>();
        float* ga_ptr = ga->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int idx = batch * r * c + i * c + j;
                    bool valid = (i < prefix_len) ? (j < prefix_len) : (j <= i);
                    ga_ptr[idx] = valid ? sg_ptr[idx] : 0.0f;
                }
            }
        }
        return {ga};
    }
};

// Applies Prefix-Causal mask: bidirectional inside prefix [0, prefix_len), causal afterwards
inline std::shared_ptr<Tensor> prefix_causal_mask(const std::shared_ptr<Tensor>& scores, int prefix_len) {
    int ndim = scores->shape.size();
    if (ndim < 2) {
        throw std::invalid_argument("prefix_causal_mask requires tensor of at least 2 dimensions");
    }
    int r = scores->shape[ndim - 2];
    int c = scores->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= scores->shape[i]; }

    bool req_grad = scores->requires_grad;
    auto out = std::make_shared<Tensor>(scores->shape, scores->device, scores->dtype, req_grad);

#ifdef USE_CUDA
    if (scores->device == Device::CUDA) {
        cuda::prefix_causal_mask(scores->data_ptr<float>(), out->data_ptr<float>(), batch_size, r, prefix_len);
        if (req_grad) {
            auto grad_fn = std::make_shared<PrefixCausalMaskBackward>(r, c, batch_size, prefix_len);
            grad_fn->add_next_edge(get_grad_edge(scores).function, 0);
            out->grad_fn = grad_fn;
        }
        return out;
    }
#endif

    const float* s_ptr = scores->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int idx = batch * r * c + i * c + j;
                bool valid = (i < prefix_len) ? (j < prefix_len) : (j <= i);
                out_ptr[idx] = valid ? s_ptr[idx] : -1e9f;
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<PrefixCausalMaskBackward>(r, c, batch_size, prefix_len);
        grad_fn->add_next_edge(get_grad_edge(scores).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// ----------------------------------------------------------------------------
// 3. ALiBi (Attention with Linear Biases) Slope Bias Matrix
//    Computes additive geometric slope bias tensor [1, n_heads, seq_len, seq_len]
//    bias[h, i, j] = -m_h * (i - j) for j <= i, -1e9 for j > i
// ----------------------------------------------------------------------------
inline std::shared_ptr<Tensor> alibi_bias(
    int n_heads, int seq_len, bool causal = true, Device device = Device::CPU
) {
    auto bias = std::make_shared<Tensor>(
        std::vector<int64_t>{1, n_heads, seq_len, seq_len}, device, DType::Float32, false
    );
    float* ptr = bias->data_ptr<float>();

    // Calculate geometric slopes m_h (Press et al. 2021)
    std::vector<float> slopes(n_heads);
    int n = 1;
    while (n * 2 <= n_heads) n *= 2;

    float base = std::pow(2.0f, -8.0f / static_cast<float>(n));
    for (int h = 0; h < n; ++h) {
        slopes[h] = std::pow(base, static_cast<float>(h + 1));
    }
    if (n < n_heads) {
        float extra_base = std::pow(2.0f, -4.0f / static_cast<float>(n));
        for (int h = 0; h < 2 * (n_heads - n); h += 2) {
            slopes[n + h / 2] = std::pow(extra_base, static_cast<float>(h + 1));
        }
    }

    for (int h = 0; h < n_heads; ++h) {
        float m = slopes[h];
        int head_offset = h * seq_len * seq_len;
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                int idx = head_offset + i * seq_len + j;
                if (causal && j > i) {
                    ptr[idx] = -1e9f;
                } else {
                    // Distance penalty: linear bias is negative for keys further away
                    ptr[idx] = -m * static_cast<float>(i - j);
                }
            }
        }
    }

    return bias;
}

// ----------------------------------------------------------------------------
// 4. Arbitrary Key-Padding Mask
//    Converts a binary/int mask [batch, seq_len] (1 = keep, 0 = pad)
//    into an additive mask [batch, 1, 1, seq_len] (0.0f = keep, -1e9f = pad)
//    Broadcasting handles [B, n_heads, seq_len, seq_len] automatically.
// ----------------------------------------------------------------------------
inline std::shared_ptr<Tensor> create_padding_mask(const std::shared_ptr<Tensor>& pad_mask) {
    if (pad_mask->shape.size() != 2) {
        throw std::invalid_argument("create_padding_mask requires pad_mask with shape [batch, seq_len]");
    }
    int batch_size = pad_mask->shape[0];
    int seq_len = pad_mask->shape[1];

    auto out = std::make_shared<Tensor>(
        std::vector<int64_t>{batch_size, 1, 1, seq_len}, pad_mask->device, DType::Float32, false
    );

    const float* in_ptr = pad_mask->data_ptr<float>();
    float* out_ptr = out->data_ptr<float>();

    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int idx = b * seq_len + s;
            out_ptr[idx] = (in_ptr[idx] > 0.5f) ? 0.0f : -1e9f;
        }
    }

    return out;
}

// ----------------------------------------------------------------------------
// 5. Document-Packing Block-Diagonal Causal Mask
//    For sequence packing / sample multiplexing in modern LLM pretraining (LLaMA 3).
//    doc_ids is [batch, seq_len] or [seq_len] specifying document index for each token.
//    Attention is allowed if and only if (doc_ids[i] == doc_ids[j] && j <= i).
// ----------------------------------------------------------------------------
inline std::shared_ptr<Tensor> document_causal_mask(const std::shared_ptr<Tensor>& doc_ids) {
    int ndim = doc_ids->ndim();
    int batch_size = (ndim == 2) ? doc_ids->shape[0] : 1;
    int seq_len = doc_ids->shape[ndim - 1];

    auto mask = std::make_shared<Tensor>(
        std::vector<int64_t>{batch_size, 1, seq_len, seq_len}, doc_ids->device, DType::Float32, false
    );

    const float* ids_ptr = doc_ids->data_ptr<float>();
    float* mask_ptr = mask->data_ptr<float>();

    for (int b = 0; b < batch_size; ++b) {
        int base_id_offset = b * seq_len;
        int base_mask_offset = b * seq_len * seq_len;

        for (int i = 0; i < seq_len; ++i) {
            float doc_i = ids_ptr[base_id_offset + i];
            for (int j = 0; j < seq_len; ++j) {
                float doc_j = ids_ptr[base_id_offset + j];
                int idx = base_mask_offset + i * seq_len + j;
                bool valid = (doc_i == doc_j) && (j <= i);
                mask_ptr[idx] = valid ? 0.0f : -1e9f;
            }
        }
    }

    return mask;
}

} // namespace masking
