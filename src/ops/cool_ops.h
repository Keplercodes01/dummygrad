#pragma once
#include "tensor.h"
#include "functional.h"

// --- SCALE AND SHIFT BACKWARD NODE ---
struct ScaleAndShiftBackward : public Node {
    std::shared_ptr<Tensor> x, g, be;
    int r, c, batch_size, ndim, nout;

    ScaleAndShiftBackward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> g, std::shared_ptr<Tensor> be,
                          int r, int c, int batch_size, int ndim, int nout)
        : x(x), g(g), be(be), r(r), c(c), batch_size(batch_size), ndim(ndim), nout(nout) {}

    void release_variables() override {
        x.reset();
        g.reset();
        be.reset();
    }

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        if (grads.empty() || !grads[0]) return {nullptr, nullptr, nullptr};
        if (!x || !g || !be) return {nullptr, nullptr, nullptr};

        std::shared_ptr<Tensor> self_grad = grads[0];
        Device dev = self_grad->device;
        auto sg_cpu = (dev == Device::CPU) ? self_grad : self_grad->cpu();
        auto x_cpu = (x->device == Device::CPU) ? x : x->cpu();
        auto g_cpu = (g->device == Device::CPU) ? g : g->cpu();
        auto be_cpu = (be->device == Device::CPU) ? be : be->cpu();

        auto gx = std::make_shared<Tensor>(x_cpu->shape, Device::CPU, x_cpu->dtype, false);
        gx->fill_(0.0f);
        auto gg = std::make_shared<Tensor>(g_cpu->shape, Device::CPU, g_cpu->dtype, false);
        gg->fill_(0.0f);
        auto gbe = std::make_shared<Tensor>(be_cpu->shape, Device::CPU, be_cpu->dtype, false);
        gbe->fill_(0.0f);

        const float* sg_ptr = sg_cpu->data_ptr<float>();
        const float* x_ptr = x_cpu->data_ptr<float>();
        const float* g_ptr = g_cpu->data_ptr<float>();

        float* gx_ptr = gx->data_ptr<float>();
        float* gg_ptr = gg->data_ptr<float>();
        float* gbe_ptr = gbe->data_ptr<float>();

        bool g_is_1d = (g_cpu->size() == c);
        bool be_is_1d = (be_cpu->size() == c);

        for (int batch = 0; batch < batch_size; batch++) {
            std::vector<int64_t> batch_idx = unravel(batch, std::vector<int64_t>(x_cpu->shape.begin(), x_cpu->shape.end() - 2));

            int batch_off_x = 0;
            int batch_off_out = 0;
            for (int i = 0; i < ndim - 2; i++) {
                batch_off_x   += batch_idx[i] * x_cpu->strides[i];
                batch_off_out += batch_idx[i] * sg_cpu->strides[i];
            }

            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int flat_x   = batch_off_x   + x_cpu->strides[ndim - 2] * i + x_cpu->strides[ndim - 1] * j;
                    int flat_out = batch_off_out  + sg_cpu->strides[nout - 2] * i + sg_cpu->strides[nout - 1] * j;
                    int g_idx    = g_is_1d ? j : (i * c + j);
                    int be_idx   = be_is_1d ? j : (i * c + j);

                    float dout = sg_ptr[flat_out];
                    gx_ptr[flat_x]  += g_ptr[g_idx] * dout;
                    gg_ptr[g_idx]   += x_ptr[flat_x] * dout;
                    gbe_ptr[be_idx] += dout;
                }
            }
        }
        return { (dev == Device::CPU) ? gx : gx->to(dev),
                 (dev == Device::CPU) ? gg : gg->to(dev),
                 (dev == Device::CPU) ? gbe : gbe->to(dev) };
    }
};

// scale_n_shift
inline std::shared_ptr<Tensor> scale_n_shift(const std::shared_ptr<Tensor>& x,
                                             const std::shared_ptr<Tensor>& gamma,
                                             const std::shared_ptr<Tensor>& beta) {
    int ndim = x->ndim();
    if (ndim < 2) {
        throw std::invalid_argument("scale_n_shift requires tensor with ndim >= 2");
    }
    int r = x->shape[ndim - 2];
    int c = x->shape[ndim - 1];
    int batch_size = 1;
    for (int i = 0; i < ndim - 2; i++) { batch_size *= x->shape[i]; }

    Device dev = x->device;
    DType dt = x->dtype;
    auto x_cpu = (dev == Device::CPU) ? x : x->cpu();
    auto g = make_contiguous((gamma->device == Device::CPU) ? gamma : gamma->cpu());
    auto be = make_contiguous((beta->device == Device::CPU) ? beta : beta->cpu());

    bool req_grad = x->requires_grad || gamma->requires_grad || beta->requires_grad;
    auto out_cpu = std::make_shared<Tensor>(x_cpu->shape, Device::CPU, dt, false);
    int nout = out_cpu->ndim();

    const float* x_ptr = x_cpu->data_ptr<float>();
    const float* g_ptr = g->data_ptr<float>();
    const float* be_ptr = be->data_ptr<float>();
    float* out_ptr = out_cpu->data_ptr<float>();

    bool g_is_1d = (g->size() == c);
    bool be_is_1d = (be->size() == c);

    for (int batch = 0; batch < batch_size; batch++) {
        std::vector<int64_t> batch_idx = unravel(batch, std::vector<int64_t>(x_cpu->shape.begin(), x_cpu->shape.end() - 2));

        int batch_off_x = 0;
        int batch_off_out = 0;
        for (int i = 0; i < ndim - 2; i++) {
            batch_off_x   += batch_idx[i] * x_cpu->strides[i];
            batch_off_out += batch_idx[i] * out_cpu->strides[i];
        }

        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                int flat_x   = batch_off_x   + x_cpu->strides[ndim - 2] * i   + x_cpu->strides[ndim - 1] * j;
                int flat_out = batch_off_out  + out_cpu->strides[nout - 2] * i + out_cpu->strides[nout - 1] * j;
                int g_idx    = g_is_1d ? j : (i * c + j);
                int be_idx   = be_is_1d ? j : (i * c + j);

                out_ptr[flat_out] = x_ptr[flat_x] * g_ptr[g_idx] + be_ptr[be_idx];
            }
        }
    }

    auto out = (dev == Device::CPU) ? out_cpu : out_cpu->to(dev);
    out->requires_grad = req_grad;

    if (req_grad) {
        auto grad_fn = std::make_shared<ScaleAndShiftBackward>(x, gamma, beta, r, c, batch_size, ndim, nout);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        grad_fn->add_next_edge(get_grad_edge(gamma).function, 1);
        grad_fn->add_next_edge(get_grad_edge(beta).function, 2);
        out->grad_fn = grad_fn;
    }

    return out;
}
