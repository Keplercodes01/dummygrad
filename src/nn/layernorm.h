#pragma once
#include "tensor.h"
#include "functional.h"
#include <cmath>


struct FusedLayerNormBackward : public Node {
    std::shared_ptr<Tensor> x, gamma, inv_std, mean, xmu;
    int r, c, batch_size;

    FusedLayerNormBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> gamma, 
                           std::shared_ptr<Tensor> inv_std, std::shared_ptr<Tensor> mean,
                           std::shared_ptr<Tensor> xmu,
                           int r, int c, int batch_size)
        : x(x), gamma(gamma), inv_std(inv_std), mean(mean), xmu(xmu), r(r), c(c), batch_size(batch_size) {}

    void release_variables() override {
        x.reset();
        gamma.reset();
        inv_std.reset();
        mean.reset();
        xmu.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0] || !x || !gamma || !inv_std || !mean || !xmu) return {nullptr, nullptr, nullptr};
        auto dout = grads[0];

#ifdef USE_CUDA
        if (x->device == Device::CUDA) {
            auto dx = std::make_shared<Tensor>(x->shape, x->device, x->dtype, false); dx->fill_(0.0f);
            auto dg = std::make_shared<Tensor>(gamma->shape, gamma->device, gamma->dtype, false); dg->fill_(0.0f);
            auto db = std::make_shared<Tensor>(gamma->shape, gamma->device, gamma->dtype, false); db->fill_(0.0f);

            cuda::layernorm_backward(dout->data_ptr<float>(), x->data_ptr<float>(), gamma->data_ptr<float>(),
                                     mean->data_ptr<float>(), inv_std->data_ptr<float>(),
                                     dx->data_ptr<float>(), dg->data_ptr<float>(), db->data_ptr<float>(),
                                     batch_size * r, c);
            return {dx, dg, db};
        }
#endif

        if (x->device != Device::CPU) {
            auto dout_cpu = dout->to(Device::CPU);
            auto x_cpu = x->to(Device::CPU);
            auto gamma_cpu = gamma->to(Device::CPU);
            auto inv_std_cpu = inv_std->to(Device::CPU);
            auto mean_cpu = mean->to(Device::CPU);
            auto xmu_cpu = xmu->to(Device::CPU);
            FusedLayerNormBackward cpu_node(x_cpu, gamma_cpu, inv_std_cpu, mean_cpu, xmu_cpu, r, c, batch_size);
            auto cpu_grads = cpu_node.apply({dout_cpu});
            if (cpu_grads.size() < 3) return {nullptr, nullptr, nullptr};
            return {
                cpu_grads[0] ? cpu_grads[0]->to(x->device) : nullptr,
                cpu_grads[1] ? cpu_grads[1]->to(gamma->device) : nullptr,
                cpu_grads[2] ? cpu_grads[2]->to(gamma->device) : nullptr
            };
        }

        auto dx = std::make_shared<Tensor>(x->shape, x->device, x->dtype, false); dx->fill_(0.0f);
        auto dg = std::make_shared<Tensor>(gamma->shape, gamma->device, gamma->dtype, false); dg->fill_(0.0f);
        auto db = std::make_shared<Tensor>(gamma->shape, gamma->device, gamma->dtype, false); db->fill_(0.0f);

        const float* dout_ptr = dout->data_ptr<float>();
        const float* xmu_ptr = xmu->data_ptr<float>();
        const float* inv_std_ptr = inv_std->data_ptr<float>();
        const float* g_ptr = gamma->data_ptr<float>();

        float* dx_ptr = dx->data_ptr<float>();
        float* dg_ptr = dg->data_ptr<float>();
        float* db_ptr = db->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                int offset = batch * r * c + i * c;
                float istd = inv_std_ptr[batch * r + i];
                
                float sum_dout = 0.0f;
                float sum_dout_xmu = 0.0f;

                for (int j = 0; j < c; j++) {
                    float d = dout_ptr[offset + j];
                    float xm = xmu_ptr[offset + j];
                    dg_ptr[j] += d * xm * istd;
                    db_ptr[j] += d;
                    
                    float dx_hat = d * g_ptr[j];
                    sum_dout += dx_hat;
                    sum_dout_xmu += dx_hat * xm;
                }

                float f_c = 1.0f / c;
                for (int j = 0; j < c; j++) {
                    float dx_hat = dout_ptr[offset + j] * g_ptr[j];
                    dx_ptr[offset + j] = istd * (dx_hat - f_c * sum_dout - f_c * xmu_ptr[offset + j] * istd * istd * sum_dout_xmu);
                }
            }
        }
        return {dx, dg, db};
    }
};

class LayerNorm {
public:
    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    float eps;

    LayerNorm(int features, float eps = 1e-5f) : eps(eps) {
        gamma = std::make_shared<Tensor>(std::vector<int64_t>{features});
        beta  = std::make_shared<Tensor>(std::vector<int64_t>{features});
        gamma->fill_(1.0f);
        beta->fill_(0.0f);
        gamma->requires_grad = true;
        beta->requires_grad = true;
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        int ndim = x->ndim();
        if (ndim < 1) throw std::runtime_error("LayerNorm: input tensor must be at least 1D");
        int c = x->shape[ndim - 1];
        if (c != gamma->shape[0]) {
            throw std::runtime_error("LayerNorm: input feature dimension mismatch, expected " +
                                     std::to_string(gamma->shape[0]) + " but got " + std::to_string(c));
        }
        int r = (ndim >= 2) ? x->shape[ndim - 2] : 1;
        int batch_size = x->size() / (r * c);

#ifdef USE_CUDA
        if (x->device == Device::CUDA) {
            if (gamma->device != Device::CUDA) gamma = gamma->to(Device::CUDA);
            if (beta->device != Device::CUDA) beta = beta->to(Device::CUDA);

            auto out = std::make_shared<Tensor>(x->shape, x->device, x->dtype, x->requires_grad);
            auto inv_std = std::make_shared<Tensor>(std::vector<int64_t>{batch_size * r}, x->device, x->dtype, false);
            auto mean = std::make_shared<Tensor>(std::vector<int64_t>{batch_size * r}, x->device, x->dtype, false);
            auto xmu = std::make_shared<Tensor>(x->shape, x->device, x->dtype, false);

            cuda::layernorm_forward(x->data_ptr<float>(), gamma->data_ptr<float>(), beta->data_ptr<float>(),
                                    out->data_ptr<float>(), mean->data_ptr<float>(), inv_std->data_ptr<float>(),
                                    batch_size * r, c, eps);

            if (x->requires_grad) {
                auto grad_fn = std::make_shared<FusedLayerNormBackward>(x, gamma, inv_std, mean, xmu, r, c, batch_size);
                grad_fn->add_next_edge(get_grad_edge(x).function, 0);
                grad_fn->add_next_edge(get_grad_edge(gamma).function, 1);
                grad_fn->add_next_edge(get_grad_edge(beta).function, 2);
                out->grad_fn = grad_fn;
            }
            return out;
        }
#endif

        if (x->device != Device::CPU) {
            auto x_cpu = x->to(Device::CPU);
            auto gamma_cpu = gamma->to(Device::CPU);
            auto beta_cpu = beta->to(Device::CPU);
            LayerNorm ln_cpu(gamma->shape[0], eps);
            ln_cpu.gamma = gamma_cpu;
            ln_cpu.beta = beta_cpu;
            auto out_cpu = ln_cpu.forward(x_cpu);
            auto out_res = out_cpu->to(x->device);
            if (x->requires_grad) {
                auto grad_fn = std::make_shared<FusedLayerNormBackward>(
                    x, gamma,
                    std::static_pointer_cast<FusedLayerNormBackward>(out_cpu->grad_fn)->inv_std->to(x->device),
                    std::static_pointer_cast<FusedLayerNormBackward>(out_cpu->grad_fn)->mean->to(x->device),
                    std::static_pointer_cast<FusedLayerNormBackward>(out_cpu->grad_fn)->xmu->to(x->device),
                    r, c, batch_size
                );
                grad_fn->add_next_edge(get_grad_edge(x).function, 0);
                grad_fn->add_next_edge(get_grad_edge(gamma).function, 1);
                grad_fn->add_next_edge(get_grad_edge(beta).function, 2);
                out_res->grad_fn = grad_fn;
            }
            return out_res;
        }

        auto out = std::make_shared<Tensor>(x->shape, x->device, x->dtype, x->requires_grad);
        auto inv_std = std::make_shared<Tensor>(std::vector<int64_t>{batch_size * r}, x->device, x->dtype, false);
        auto mean = std::make_shared<Tensor>(std::vector<int64_t>{batch_size * r}, x->device, x->dtype, false);
        auto xmu = std::make_shared<Tensor>(x->shape, x->device, x->dtype, false);

        const float* x_ptr = x->data_ptr<float>();
        const float* g_ptr = gamma->data_ptr<float>();
        const float* b_ptr = beta->data_ptr<float>();
        float* out_ptr = out->data_ptr<float>();
        float* xmu_ptr = xmu->data_ptr<float>();
        float* istd_ptr = inv_std->data_ptr<float>();
        float* mean_ptr = mean->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                int offset = batch * r * c + i * c;
                
                float m = 0.0f;
                for (int j = 0; j < c; j++) m += x_ptr[offset + j];
                m /= c;
                mean_ptr[batch * r + i] = m;

                float var = 0.0f;
                for (int j = 0; j < c; j++) {
                    float diff = x_ptr[offset + j] - m;
                    xmu_ptr[offset + j] = diff;
                    var += diff * diff;
                }
                var /= c;

                float istd = 1.0f / std::sqrt(var + eps);
                istd_ptr[batch * r + i] = istd;

                for (int j = 0; j < c; j++) {
                    out_ptr[offset + j] = xmu_ptr[offset + j] * istd * g_ptr[j] + b_ptr[j];
                }
            }
        }

        if (x->requires_grad) {
            auto grad_fn = std::make_shared<FusedLayerNormBackward>(x, gamma, inv_std, mean, xmu, r, c, batch_size);
            grad_fn->add_next_edge(get_grad_edge(x).function, 0);
            grad_fn->add_next_edge(get_grad_edge(gamma).function, 1);
            grad_fn->add_next_edge(get_grad_edge(beta).function, 2);
            out->grad_fn = grad_fn;
        }

        return out;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const { return {gamma, beta}; }

    void to(Device device, DType dtype = DType::Float32) {
        gamma = gamma->to(device, dtype);
        beta = beta->to(device, dtype);
        gamma->requires_grad = true;
        beta->requires_grad = true;
    }
};
