#pragma once
#include "tensor.h"
#include "functional.h"
#include <cmath>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

struct FusedLayerNormBackward : public Node {
    std::shared_ptr<Tensor> x, gamma, inv_std, xmu;
    int r, c, batch_size;

    FusedLayerNormBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> gamma, 
                           std::shared_ptr<Tensor> inv_std, std::shared_ptr<Tensor> xmu,
                           int r, int c, int batch_size)
        : x(x), gamma(gamma), inv_std(inv_std), xmu(xmu), r(r), c(c), batch_size(batch_size) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        auto dout = grads[0];
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
        int c = x->shape[ndim - 1];
        int r = (ndim >= 2) ? x->shape[ndim - 2] : 1;
        int batch_size = x->size() / (r * c);

        auto out = std::make_shared<Tensor>(x->shape, x->device, x->dtype, x->requires_grad);
        auto inv_std = std::make_shared<Tensor>(std::vector<int64_t>{batch_size * r}, x->device, x->dtype, false);
        auto xmu = std::make_shared<Tensor>(x->shape, x->device, x->dtype, false);

        const float* x_ptr = x->data_ptr<float>();
        const float* g_ptr = gamma->data_ptr<float>();
        const float* b_ptr = beta->data_ptr<float>();
        float* out_ptr = out->data_ptr<float>();
        float* xmu_ptr = xmu->data_ptr<float>();
        float* istd_ptr = inv_std->data_ptr<float>();

        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < r; i++) {
                int offset = batch * r * c + i * c;
                
                float mean = 0.0f;
#if defined(__APPLE__)
                vDSP_sve(x_ptr + offset, 1, &mean, c);
                mean /= c;
                
                float neg_mean = -mean;
                vDSP_vsadd(x_ptr + offset, 1, &neg_mean, xmu_ptr + offset, 1, c);
                
                float var = 0.0f;
                vDSP_measqv(xmu_ptr + offset, 1, &var, c);
#else
                for (int j = 0; j < c; j++) mean += x_ptr[offset + j];
                mean /= c;
                float var = 0.0f;
                for (int j = 0; j < c; j++) {
                    xmu_ptr[offset + j] = x_ptr[offset + j] - mean;
                    var += xmu_ptr[offset + j] * xmu_ptr[offset + j];
                }
                var /= c;
#endif
                float istd = 1.0f / std::sqrt(var + eps);
                istd_ptr[batch * r + i] = istd;

#if defined(__APPLE__)
                vDSP_vsmul(xmu_ptr + offset, 1, &istd, out_ptr + offset, 1, c);
                vDSP_vmul(out_ptr + offset, 1, g_ptr, 1, out_ptr + offset, 1, c);
                vDSP_vadd(out_ptr + offset, 1, b_ptr, 1, out_ptr + offset, 1, c);
#else
                for (int j = 0; j < c; j++) {
                    out_ptr[offset + j] = xmu_ptr[offset + j] * istd * g_ptr[j] + b_ptr[j];
                }
#endif
            }
        }

        if (x->requires_grad) {
            auto grad_fn = std::make_shared<FusedLayerNormBackward>(x, gamma, inv_std, xmu, r, c, batch_size);
            grad_fn->add_next_edge(get_grad_edge(x).function, 0);
            grad_fn->add_next_edge(get_grad_edge(gamma).function, 1);
            grad_fn->add_next_edge(get_grad_edge(beta).function, 2);
            out->grad_fn = grad_fn;
        }

        return out;
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const { return {gamma, beta}; }
};
