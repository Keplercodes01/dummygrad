#pragma once
#include "engine.h"

// --- BROADCAST BACKWARD NODE ---
struct BroadcastBackward : public Node {
    std::vector<int> in_shape;
    int axis, n, r, c, batch_size;

    BroadcastBackward(const std::vector<int>& in_shape, int axis, int n, int r, int c, int batch_size)
        : in_shape(in_shape), axis(axis), n(n), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto grad_a = std::make_shared<Tensor>(in_shape, false);
        const float* sg_ptr = self_grad->data_ptr();
        float* ga_ptr = grad_a->data_ptr();

        if (axis == 0) {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < c; j++) {
                        ga_ptr[j + batch * c] += sg_ptr[i * c + j + batch * n * c];
                    }
                }
            }
        } else {
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < r; i++) {
                    for (int j = 0; j < n; j++) {
                        ga_ptr[i + batch * r] += sg_ptr[i * n + j + batch * r * n];
                    }
                }
            }
        }
        return {grad_a};
    }
};

// broadcast
inline std::shared_ptr<Tensor> broadcast(const std::shared_ptr<Tensor>& a, int axis, int n) {
    int ndim = a->shape.size();
    int r = a->shape[ndim - 2];
    int c = a->shape[ndim - 1];

    if (axis == 0 && r != 1) { throw std::runtime_error("The dimension to be broadcasted should be 1...cmon man"); }
    if (axis == 1 && c != 1) { throw std::runtime_error("The dimension to be broadcasted should be 1...cmon man"); }

    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= a->shape[i]; } 

    std::vector<int> out_shape = a->shape;
    axis == 0 ? out_shape[ndim - 2] = n : out_shape[ndim - 1] = n;

    bool req_grad = a->requires_grad;
    auto out = std::make_shared<Tensor>(out_shape, req_grad);
    const float* a_ptr = a->data_ptr();
    float* out_ptr = out->data_ptr();

    if (axis == 0) {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < c; j++) {
                    out_ptr[i * c + j + batch * n * c] = a_ptr[j + batch * c];
                }
            }
        }
    } else {
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < n; j++) {
                    out_ptr[i * n + j + batch * r * n] = a_ptr[i + batch * r];
                }
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<BroadcastBackward>(a->shape, axis, n, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(a).function, 0);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- CAST ADD BACKWARD NODE ---
struct CastAddBackward : public Node {
    std::vector<int> x_shape, y_shape;
    int r, c, batch_size;

    CastAddBackward(const std::vector<int>& x_shape, const std::vector<int>& y_shape, int r, int c, int batch_size)
        : x_shape(x_shape), y_shape(y_shape), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto gx = std::make_shared<Tensor>(x_shape, false);
        auto gy = std::make_shared<Tensor>(y_shape, false);

        const float* sg_ptr = self_grad->data_ptr();
        float* gx_ptr = gx->data_ptr();
        float* gy_ptr = gy->data_ptr();

        int yr = y_shape.size() > 1 ? y_shape[y_shape.size() - 2] : 1;
        int yc = y_shape.size() > 0 ? y_shape[y_shape.size() - 1] : 1;

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int x_idx = batch * r * c + i * c + j;
                    int yi = (yr == 1) ? 0 : i;
                    int yj = (yc == 1) ? 0 : j;
                    int y_idx = yi * yc + yj;

                    float val = sg_ptr[x_idx];
                    gx_ptr[x_idx] += val;
                    gy_ptr[y_idx] += val;
                }
            }
        }
        return {gx, gy};
    }
};

// broadcast and add
inline std::shared_ptr<Tensor> cast_n_add(const std::shared_ptr<Tensor>& x,
                                          const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = x->shape[ndim - 2];
    int c = x->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= x->shape[i]; }

    int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
    int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

    bool req_grad = x->requires_grad || y->requires_grad;
    auto out = std::make_shared<Tensor>(x->shape, req_grad);
    const float* x_ptr = x->data_ptr();
    const float* y_ptr = y->data_ptr();
    float* out_ptr = out->data_ptr();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int x_idx = batch * r * c + i * c + j;
                int yi = (yr == 1) ? 0 : i;
                int yj = (yc == 1) ? 0 : j;
                int y_idx = yi * yc + yj;
                out_ptr[x_idx] = x_ptr[x_idx] + y_ptr[y_idx];
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<CastAddBackward>(x->shape, y->shape, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        grad_fn->add_next_edge(get_grad_edge(y).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- CAST SUB BACKWARD NODE ---
struct CastSubBackward : public Node {
    std::vector<int> x_shape, y_shape;
    int r, c, batch_size;

    CastSubBackward(const std::vector<int>& x_shape, const std::vector<int>& y_shape, int r, int c, int batch_size)
        : x_shape(x_shape), y_shape(y_shape), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto gx = std::make_shared<Tensor>(x_shape, false);
        auto gy = std::make_shared<Tensor>(y_shape, false);

        const float* sg_ptr = self_grad->data_ptr();
        float* gx_ptr = gx->data_ptr();
        float* gy_ptr = gy->data_ptr();

        int yr = y_shape.size() > 1 ? y_shape[y_shape.size() - 2] : 1;
        int yc = y_shape.size() > 0 ? y_shape[y_shape.size() - 1] : 1;

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int x_idx = batch * r * c + i * c + j;
                    int yi = (yr == 1) ? 0 : i;
                    int yj = (yc == 1) ? 0 : j;
                    int y_idx = yi * yc + yj;

                    float val = sg_ptr[x_idx];
                    gx_ptr[x_idx] += val;
                    gy_ptr[y_idx] -= val;
                }
            }
        }
        return {gx, gy};
    }
};

// broadcast and subtract
inline std::shared_ptr<Tensor> cast_n_sub(const std::shared_ptr<Tensor>& x,
                                          const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = x->shape[ndim - 2];
    int c = x->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= x->shape[i]; }

    int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
    int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

    bool req_grad = x->requires_grad || y->requires_grad;
    auto out = std::make_shared<Tensor>(x->shape, req_grad);
    const float* x_ptr = x->data_ptr();
    const float* y_ptr = y->data_ptr();
    float* out_ptr = out->data_ptr();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int x_idx = batch * r * c + i * c + j;
                int yi = (yr == 1) ? 0 : i;
                int yj = (yc == 1) ? 0 : j;
                int y_idx = yi * yc + yj;
                out_ptr[x_idx] = x_ptr[x_idx] - y_ptr[y_idx];
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<CastSubBackward>(x->shape, y->shape, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        grad_fn->add_next_edge(get_grad_edge(y).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- CAST MUL BACKWARD NODE ---
struct CastMulBackward : public Node {
    std::shared_ptr<Tensor> x, y;
    int r, c, batch_size;

    CastMulBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y, int r, int c, int batch_size)
        : x(x), y(y), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto gx = std::make_shared<Tensor>(x->shape, false);
        auto gy = std::make_shared<Tensor>(y->shape, false);

        const float* sg_ptr = self_grad->data_ptr();
        const float* x_ptr = x->data_ptr();
        const float* y_ptr = y->data_ptr();
        float* gx_ptr = gx->data_ptr();
        float* gy_ptr = gy->data_ptr();

        int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
        int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int x_idx = batch * r * c + i * c + j;
                    int yi = (yr == 1) ? 0 : i;
                    int yj = (yc == 1) ? 0 : j;
                    int y_idx = yi * yc + yj;

                    float val = sg_ptr[x_idx];
                    gx_ptr[x_idx] += y_ptr[y_idx] * val;
                    gy_ptr[y_idx] += x_ptr[x_idx] * val;
                }
            }
        }
        return {gx, gy};
    }
};

// broadcast and multiply
inline std::shared_ptr<Tensor> cast_n_mul(const std::shared_ptr<Tensor>& x,
                                          const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = x->shape[ndim - 2];
    int c = x->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= x->shape[i]; }

    int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
    int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

    bool req_grad = x->requires_grad || y->requires_grad;
    auto out = std::make_shared<Tensor>(x->shape, req_grad);
    const float* x_ptr = x->data_ptr();
    const float* y_ptr = y->data_ptr();
    float* out_ptr = out->data_ptr();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int x_idx = batch * r * c + i * c + j;
                int yi = (yr == 1) ? 0 : i;
                int yj = (yc == 1) ? 0 : j;
                int y_idx = yi * yc + yj;
                out_ptr[x_idx] = x_ptr[x_idx] * y_ptr[y_idx];
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<CastMulBackward>(x, y, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        grad_fn->add_next_edge(get_grad_edge(y).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}

// --- CAST DIV BACKWARD NODE ---
struct CastDivBackward : public Node {
    std::shared_ptr<Tensor> x, y;
    int r, c, batch_size;

    CastDivBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> y, int r, int c, int batch_size)
        : x(x), y(y), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        std::shared_ptr<Tensor> self_grad = grads[0];
        auto gx = std::make_shared<Tensor>(x->shape, false);
        auto gy = std::make_shared<Tensor>(y->shape, false);

        const float* sg_ptr = self_grad->data_ptr();
        const float* x_ptr = x->data_ptr();
        const float* y_ptr = y->data_ptr();
        float* gx_ptr = gx->data_ptr();
        float* gy_ptr = gy->data_ptr();

        int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
        int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int x_idx = batch * r * c + i * c + j;
                    int yi = (yr == 1) ? 0 : i;
                    int yj = (yc == 1) ? 0 : j;
                    int y_idx = yi * yc + yj;

                    float val = sg_ptr[x_idx];
                    float y_val = y_ptr[y_idx];
                    gx_ptr[x_idx] += val / y_val;
                    gy_ptr[y_idx] -= (x_ptr[x_idx] * val) / (y_val * y_val);
                }
            }
        }
        return {gx, gy};
    }
};

// broadcast and divide
inline std::shared_ptr<Tensor> cast_n_div(const std::shared_ptr<Tensor>& x,
                                          const std::shared_ptr<Tensor>& y) {
    int ndim = x->shape.size();
    int r = x->shape[ndim - 2];
    int c = x->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= x->shape[i]; }

    int yr = y->shape.size() > 1 ? y->shape[y->shape.size() - 2] : 1;
    int yc = y->shape.size() > 0 ? y->shape[y->shape.size() - 1] : 1;

    bool req_grad = x->requires_grad || y->requires_grad;
    auto out = std::make_shared<Tensor>(x->shape, req_grad);
    const float* x_ptr = x->data_ptr();
    const float* y_ptr = y->data_ptr();
    float* out_ptr = out->data_ptr();

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int x_idx = batch * r * c + i * c + j;
                int yi = (yr == 1) ? 0 : i;
                int yj = (yc == 1) ? 0 : j;
                int y_idx = yi * yc + yj;
                out_ptr[x_idx] = x_ptr[x_idx] / y_ptr[y_idx];
            }
        }
    }

    if (req_grad) {
        auto grad_fn = std::make_shared<CastDivBackward>(x, y, r, c, batch_size);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        grad_fn->add_next_edge(get_grad_edge(y).function, 1);
        out->grad_fn = grad_fn;
    }

    return out;
}
