#pragma once
#include "engine.h"

// --- MATMUL BACKWARD NODE ---
struct MatmulBackward : public Node {
    std::shared_ptr<Tensor> a, b;
    int r1, c1, r2, c2, batch_size, n1, n2, nout, nout_batch;
    std::vector<int> strides_a_broad, strides_b_broad, out_shape;

    MatmulBackward(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                   int r1, int c1, int r2, int c2, int batch_size,
                   int n1, int n2, int nout, int nout_batch,
                   std::vector<int> strides_a_broad, std::vector<int> strides_b_broad,
                   std::vector<int> out_shape)
        : a(a), b(b), r1(r1), c1(c1), r2(r2), c2(c2), batch_size(batch_size),
          n1(n1), n2(n2), nout(nout), nout_batch(nout_batch),
          strides_a_broad(std::move(strides_a_broad)),
          strides_b_broad(std::move(strides_b_broad)),
          out_shape(std::move(out_shape)) {}

    void release_variables() override {
        a = nullptr;
        b = nullptr;
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];

        auto grad_a = std::make_shared<Tensor>(a->shape, false);
        auto grad_b = std::make_shared<Tensor>(b->shape, false);

        const float* b_data = b->data_ptr();
        const float* a_data = a->data_ptr();
        const float* out_grad_data = self_grad->data_ptr();

        float* ga_data = grad_a->data_ptr();
        float* gb_data = grad_b->data_ptr();

        for (int batch = 0; batch < batch_size; batch++) {
            std::vector<int> batch_idx = unravel(batch, std::vector<int>(out_shape.begin(), out_shape.begin() + nout_batch)); 

            int batch_off_a = 0;
            int batch_off_b = 0;
            int batch_off_out = 0;

            for (int i = 0; i < nout_batch; i++) {
                batch_off_a += batch_idx[i] * strides_a_broad[i];
                batch_off_b += batch_idx[i] * strides_b_broad[i];
                batch_off_out += batch_idx[i] * self_grad->strides[i];
            }

            // a.grad computation
            for (int i = 0; i < r1; i++) {
                for (int j = 0; j < c1; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < c2; k++) {
                        int flat_b = batch_off_b + b->strides[n2-2]*j + b->strides[n2-1]*k;
                        int flat_out = batch_off_out + self_grad->strides[nout-2]*i + self_grad->strides[nout-1]*k;
                        sum += out_grad_data[flat_out] * b_data[flat_b];
                    }
                    int flat_a = batch_off_a + a->strides[n1-2]*i + a->strides[n1-1]*j;
                    ga_data[flat_a] += sum;
                }
            }

            // b.grad computation
            for (int i = 0; i < r2; i++) {
                for (int j = 0; j < c2; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < r1; k++) {
                        int flat_a = batch_off_a + a->strides[n1-2]*k + a->strides[n1-1]*i;
                        int flat_out = batch_off_out + self_grad->strides[nout-2]*k + self_grad->strides[nout-1]*j;
                        sum += a_data[flat_a] * out_grad_data[flat_out];
                    }
                    int flat_b = batch_off_b + b->strides[n2-2]*i + b->strides[n2-1]*j;
                    gb_data[flat_b] += sum;
                }    
            }
        }

        return {grad_a, grad_b};
    }
};

// manual matmul forward function
inline std::shared_ptr<Tensor> manual_matmul(const std::shared_ptr<Tensor>& _a, const std::shared_ptr<Tensor>& _b) {
    auto a = make_contiguous(_a);
    auto b = make_contiguous(_b);

    int n1 = a->ndim();
    int n2 = b->ndim();

    if (n1 < 2 || n2 < 2) {
        throw std::runtime_error("matmul: Tensors must be at least 2D...");
    }

    int r1 = a->shape[n1 - 2];
    int c1 = a->shape[n1 - 1];
    int r2 = b->shape[n2 - 2];
    int c2 = b->shape[n2 - 1];

    if (c1 != r2) {
        throw std::runtime_error("matmul: inner dimensions mismatch...");  
    }

    int b1 = n1 - 2;
    int b2 = n2 - 2;
    int nout_batch = std::max(b1, b2);

    std::vector<int> out_shape;
    std::vector<int> strides_a_broad(nout_batch, 0);
    std::vector<int> strides_b_broad(nout_batch, 0);

    for (int i = 0; i < nout_batch; ++i) {
        int idx_a = i - (nout_batch - b1);
        int idx_b = i - (nout_batch - b2);

        int dim_a = (idx_a >= 0) ? a->shape[idx_a] : 1;
        int dim_b = (idx_b >= 0) ? b->shape[idx_b] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::runtime_error("matmul: batch dimensions mismatch for broadcasting...");
        }

        int dim_out = std::max(dim_a, dim_b);
        out_shape.push_back(dim_out);

        if (idx_a >= 0 && dim_a != 1) {
            strides_a_broad[i] = a->strides[idx_a];
        } else {
            strides_a_broad[i] = 0;
        }

        if (idx_b >= 0 && dim_b != 1) {
            strides_b_broad[i] = b->strides[idx_b];
        } else {
            strides_b_broad[i] = 0;
        }
    }

    out_shape.push_back(r1);
    out_shape.push_back(c2);

    bool req_grad = _a->requires_grad || _b->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);
    int nout = out->ndim();

    int batch_size = 1;
    for (int i = 0; i < nout_batch; i++) {
        batch_size *= out_shape[i];
    }

    const float* a_data = a->data_ptr();
    const float* b_data = b->data_ptr();
    float* out_data = out->data_ptr();

    // Forward pass
    for (int batch = 0; batch < batch_size; batch++) {
        std::vector<int> batch_idx = unravel(batch, std::vector<int>(out_shape.begin(), out_shape.begin() + nout_batch)); 

        int batch_off_a = 0;
        int batch_off_b = 0;
        int batch_off_out = 0;

        for (int i = 0; i < nout_batch; i++) {
            batch_off_a += batch_idx[i] * strides_a_broad[i];
            batch_off_b += batch_idx[i] * strides_b_broad[i];
            batch_off_out += batch_idx[i] * out->strides[i];
        }
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                float sum = 0.0f;
                for (int k = 0; k < c1; k++) {
                    int flat_a = batch_off_a + a->strides[n1-2]*i + a->strides[n1-1]*k;
                    int flat_b = batch_off_b + b->strides[n2-2]*k + b->strides[n2-1]*j;
                    sum += a_data[flat_a] * b_data[flat_b];
                }
                int flat_out = batch_off_out + out->strides[nout-2]*i + out->strides[nout-1]*j;
                out_data[flat_out] = sum; 
            }
        }
    }

    // Attach Autograd Node if needed
    if (req_grad) {
        auto grad_fn = std::make_shared<MatmulBackward>(
            a, b, r1, c1, r2, c2, batch_size, n1, n2, nout, nout_batch,
            strides_a_broad, strides_b_broad, out_shape
        );
        grad_fn->add_next_edge(get_grad_edge(_a).function, 0);
        grad_fn->add_next_edge(get_grad_edge(_b).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

inline std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return manual_matmul(a, b);
}
