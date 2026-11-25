#pragma once
// Layers: containers, activations, dropout, convolutions, pooling, normalization, upsampling. All are
// Modules; names match torch.nn where the layer does.

#include "functional.h"
#include "init.h"
#include "module.h"
#include "linear.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------------------------
// Initialization (in place, on parameters of any shape; fan_in / fan_out follow PyTorch's convention
// for weights [out, in, ...])
// ---------------------------------------------------------------------------------------------

namespace init {

inline void set_values(const std::shared_ptr<Tensor>& t, const std::shared_ptr<Tensor>& values) {
    NoGradGuard no_grad;
    ops::make_contiguous(*t, *values->to(t->device, t->dtype));
}
inline std::pair<int64_t, int64_t> fans(const Tensor& t) {
    if (t.ndim() < 2) throw std::invalid_argument("init: fan in and out need at least two dimensions");
    int64_t field = 1;
    for (size_t d = 2; d < t.shape.size(); ++d) field *= t.shape[d];
    return {t.shape[1] * field, t.shape[0] * field};
}
inline void constant_(const std::shared_ptr<Tensor>& t, float v) { t->fill_(v); }
inline void zeros_(const std::shared_ptr<Tensor>& t) { t->fill_(0.0f); }
inline void ones_(const std::shared_ptr<Tensor>& t) { t->fill_(1.0f); }
inline void uniform_(const std::shared_ptr<Tensor>& t, float lo = 0.0f, float hi = 1.0f) { set_values(t, uniform(t->shape, lo, hi, t->device)); }
inline void normal_(const std::shared_ptr<Tensor>& t, float mean = 0.0f, float std = 1.0f) { set_values(t, normal(t->shape, mean, std, t->device)); }
// Normal cut off outside [a, b], by the inverse CDF
inline void trunc_normal_(const std::shared_ptr<Tensor>& t, float mean = 0.0f, float std = 1.0f, float a = -2.0f, float b = 2.0f) {
    NoGradGuard no_grad;
    auto cdf = [](double x) { return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0))); };
    double lo = cdf((a - mean) / std), hi = cdf((b - mean) / std);
    auto u = uniform(t->shape, (float)(2.0 * lo - 1.0), (float)(2.0 * hi - 1.0), t->device);
    auto v = add_scalar(mul_scalar(erfinv(clamp(u, -0.9999999f, 0.9999999f)), std * 1.41421356237309505f), mean);
    set_values(t, clamp(v, a, b));
}
inline void xavier_uniform_(const std::shared_ptr<Tensor>& t, float gain = 1.0f) {
    auto f = fans(*t);
    float a = gain * std::sqrt(6.0f / (float)(f.first + f.second));
    uniform_(t, -a, a);
}
inline void xavier_normal_(const std::shared_ptr<Tensor>& t, float gain = 1.0f) {
    auto f = fans(*t);
    normal_(t, 0.0f, gain * std::sqrt(2.0f / (float)(f.first + f.second)));
}
// slope is the negative slope of the leaky relu that follows (0 for relu)
inline void kaiming_uniform_(const std::shared_ptr<Tensor>& t, float slope = 0.0f, bool fan_out = false) {
    auto f = fans(*t);
    float a = std::sqrt(2.0f / (1.0f + slope * slope)) * std::sqrt(3.0f / (float)(fan_out ? f.second : f.first));
    uniform_(t, -a, a);
}
inline void kaiming_normal_(const std::shared_ptr<Tensor>& t, float slope = 0.0f, bool fan_out = false) {
    auto f = fans(*t);
    normal_(t, 0.0f, std::sqrt(2.0f / (1.0f + slope * slope)) / std::sqrt((float)(fan_out ? f.second : f.first)));
}
inline void eye_(const std::shared_ptr<Tensor>& t) { set_values(t, eye(t->shape[0], t->shape[1])); }

}  // namespace init

// ---------------------------------------------------------------------------------------------
// Containers
// ---------------------------------------------------------------------------------------------

// A module with the usual forward(x), so it can go into a Sequential
class Layer : public Module {
public:
    virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) = 0;
    std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& x) { return forward(x); }
    void visit(ModuleVisitor&) override {}
};

// Layers applied in order. Elements are Layers, modules with forward(x), or plain functions:
//
//   auto net = std::make_shared<Sequential>();
//   net->add<Linear>(784, 256).add<ReLU>().add<Dropout>(0.1f).add<Linear>(256, 10);
//   net->add([](const TensorPtr& x) { return softmax(x); });
class Sequential : public Layer {
public:
    Sequential() = default;

    template <typename M, typename... Args>
    Sequential& add(Args&&... args) { return add(std::make_shared<M>(std::forward<Args>(args)...)); }

    template <typename M, typename = std::enable_if_t<std::is_base_of_v<Module, M>>>
    Sequential& add(std::shared_ptr<M> m) {
        Item item;
        item.module = m;
        item.run = [m](const std::shared_ptr<Tensor>& x) { return m->forward(x); };
        items_.push_back(std::move(item));
        return *this;
    }
    Sequential& add(std::function<std::shared_ptr<Tensor>(const std::shared_ptr<Tensor>&)> fn) {
        items_.push_back({nullptr, std::move(fn)});
        return *this;
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override {
        auto y = x;
        for (auto& item : items_) y = item.run(y);
        return y;
    }
    size_t size() const { return items_.size(); }
    void visit(ModuleVisitor& v) override {
        for (size_t i = 0; i < items_.size(); ++i) {
            if (items_[i].module) v.module(std::to_string(i), *items_[i].module);
        }
    }

private:
    struct Item {
        std::shared_ptr<Module> module;  // null for plain functions
        std::function<std::shared_ptr<Tensor>(const std::shared_ptr<Tensor>&)> run;
    };
    std::vector<Item> items_;
};

// A list of submodules of one type, named by position
template <typename M>
class ModuleList : public Module {
public:
    template <typename... Args>
    M& emplace_back(Args&&... args) {
        items_.push_back(std::make_shared<M>(std::forward<Args>(args)...));
        return *items_.back();
    }
    void push_back(std::shared_ptr<M> m) { items_.push_back(std::move(m)); }
    M& operator[](size_t i) { return *items_.at(i); }
    size_t size() const { return items_.size(); }
    auto begin() { return items_.begin(); }
    auto end() { return items_.end(); }
    void visit(ModuleVisitor& v) override {
        for (size_t i = 0; i < items_.size(); ++i) v.module(std::to_string(i), *items_[i]);
    }

private:
    std::vector<std::shared_ptr<M>> items_;
};

// ---------------------------------------------------------------------------------------------
// Activations and reshaping
// ---------------------------------------------------------------------------------------------

#define DG_ACTIVATION_LAYER(Name, expr)                                                    \
    class Name : public Layer {                                                            \
    public:                                                                                \
        std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return expr; } \
    }

DG_ACTIVATION_LAYER(Identity, x);
DG_ACTIVATION_LAYER(ReLU, relu(x));
DG_ACTIVATION_LAYER(ReLU6, relu6(x));
DG_ACTIVATION_LAYER(GELU, gelu_erf(x));
DG_ACTIVATION_LAYER(GELUTanh, gelu(x));
DG_ACTIVATION_LAYER(SiLU, silu(x));
DG_ACTIVATION_LAYER(Mish, mish(x));
DG_ACTIVATION_LAYER(Tanh, tanh(x));
DG_ACTIVATION_LAYER(Sigmoid, sigmoid(x));
DG_ACTIVATION_LAYER(LogSigmoid, log_sigmoid(x));
DG_ACTIVATION_LAYER(SELU, selu(x));
DG_ACTIVATION_LAYER(Hardswish, hardswish(x));
DG_ACTIVATION_LAYER(Hardsigmoid, hardsigmoid(x));
DG_ACTIVATION_LAYER(Softsign, softsign(x));
DG_ACTIVATION_LAYER(Tanhshrink, tanhshrink(x));
#undef DG_ACTIVATION_LAYER

class LeakyReLU : public Layer {
public:
    float slope;
    explicit LeakyReLU(float slope = 0.01f) : slope(slope) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return leaky_relu(x, slope); }
};
class ELU : public Layer {
public:
    float alpha;
    explicit ELU(float alpha = 1.0f) : alpha(alpha) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return elu(x, alpha); }
};
class Softplus : public Layer {
public:
    float beta;
    explicit Softplus(float beta = 1.0f) : beta(beta) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return softplus(x, beta); }
};
class Softmax : public Layer {
public:
    int dim;
    explicit Softmax(int dim = -1) : dim(dim) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return softmax(x, dim); }
};
class LogSoftmax : public Layer {
public:
    int dim;
    explicit LogSoftmax(int dim = -1) : dim(dim) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return log_softmax(x, dim); }
};
class PReLU : public Layer {
public:
    std::shared_ptr<Tensor> weight;
    explicit PReLU(int num_parameters = 1, float slope = 0.25f) { weight = tensor(std::vector<float>((size_t)num_parameters, slope), Device::CPU, true); }
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return prelu(x, weight); }
    void visit(ModuleVisitor& v) override { v.parameter("weight", weight); }
};
// Splits dim in two and gates one half with the other's sigmoid
class GLU : public Layer {
public:
    int dim;
    explicit GLU(int dim = -1) : dim(dim) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override {
        auto parts = chunk(x, 2, dim);
        return mul(parts[0], sigmoid(parts[1]));
    }
};
class Flatten : public Layer {
public:
    int start, end;
    explicit Flatten(int start = 1, int end = -1) : start(start), end(end) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return flatten(x, start, end); }
};
class Unflatten : public Layer {
public:
    int dim;
    std::vector<int64_t> sizes;
    Unflatten(int dim, std::vector<int64_t> sizes) : dim(dim), sizes(std::move(sizes)) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return unflatten(x, dim, sizes); }
};

// ---------------------------------------------------------------------------------------------
// Dropout: active in train mode only
// ---------------------------------------------------------------------------------------------

class Dropout : public Layer {
public:
    float p;
    explicit Dropout(float p = 0.5f) : p(p) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return dropout(x, p, training_); }
};
// Whole channels
class FeatureDropout : public Layer {
public:
    float p;
    explicit FeatureDropout(float p = 0.5f) : p(p) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return feature_dropout(x, p, training_); }
};
using Dropout1d = FeatureDropout;
using Dropout2d = FeatureDropout;
using Dropout3d = FeatureDropout;
class AlphaDropout : public Layer {
public:
    float p;
    explicit AlphaDropout(float p = 0.5f) : p(p) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return alpha_dropout(x, p, training_); }
};
class DropPath : public Layer {
public:
    float p;
    explicit DropPath(float p) : p(p) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return drop_path(x, p, training_); }
};

// ---------------------------------------------------------------------------------------------
// Convolutions
// ---------------------------------------------------------------------------------------------

struct ConvOptions {
    std::vector<int64_t> stride{1}, padding{0}, dilation{1}, output_padding{0};
    int64_t groups = 1;
    bool bias = true;
    PadMode pad_mode = PadMode::Constant;
};

// N spatial dimensions
template <int N>
class ConvNd : public Layer {
public:
    std::shared_ptr<Tensor> weight, bias;
    ConvOptions options;

    ConvNd(int64_t in, int64_t out, std::vector<int64_t> kernel, ConvOptions opt = {}) : options(std::move(opt)) {
        if (options.groups < 1 || in % options.groups != 0 || out % options.groups != 0) throw std::invalid_argument("Conv: groups must divide both channel counts");
        auto k = conv_detail::spread(kernel, N, "Conv");
        std::vector<int64_t> shape{out, in / options.groups};
        shape.insert(shape.end(), k.begin(), k.end());
        weight = std::make_shared<Tensor>(shape, true);
        init::kaiming_uniform_(weight, std::sqrt(5.0f));
        if (options.bias) {
            bias = std::make_shared<Tensor>(std::vector<int64_t>{out}, true);
            float bound = 1.0f / std::sqrt((float)init::fans(*weight).first);
            init::uniform_(bias, -bound, bound);
        }
    }
    // Square kernel with stride and padding, the common case
    ConvNd(int64_t in, int64_t out, int64_t kernel, int64_t stride = 1, int64_t padding = 0, int64_t groups = 1, bool use_bias = true, int64_t dilation = 1)
        : ConvNd(in, out, std::vector<int64_t>{kernel}, make_options(stride, padding, groups, use_bias, dilation)) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override {
        if (options.pad_mode == PadMode::Constant) return conv_nd(x, weight, bias, options.stride, options.padding, options.dilation, options.groups);
        auto p = conv_detail::spread(options.padding, N, "Conv");
        std::vector<int64_t> amounts;
        for (int d = N - 1; d >= 0; --d) {
            amounts.push_back(p[(size_t)d]);
            amounts.push_back(p[(size_t)d]);
        }
        return conv_nd(pad(x, amounts, options.pad_mode), weight, bias, options.stride, {0}, options.dilation, options.groups);
    }
    void visit(ModuleVisitor& v) override {
        v.parameter("weight", weight);
        if (bias) v.parameter("bias", bias);
    }

    static ConvOptions make_options(int64_t stride, int64_t padding, int64_t groups, bool use_bias, int64_t dilation) {
        ConvOptions o;
        o.stride = {stride};
        o.padding = {padding};
        o.dilation = {dilation};
        o.groups = groups;
        o.bias = use_bias;
        return o;
    }
};
using Conv1d = ConvNd<1>;
using Conv2d = ConvNd<2>;
using Conv3d = ConvNd<3>;

template <int N>
class ConvTransposeNd : public Layer {
public:
    std::shared_ptr<Tensor> weight, bias;
    ConvOptions options;

    ConvTransposeNd(int64_t in, int64_t out, std::vector<int64_t> kernel, ConvOptions opt = {}) : options(std::move(opt)) {
        if (options.groups < 1 || in % options.groups != 0 || out % options.groups != 0) throw std::invalid_argument("ConvTranspose: groups must divide both channel counts");
        auto k = conv_detail::spread(kernel, N, "ConvTranspose");
        std::vector<int64_t> shape{in, out / options.groups};
        shape.insert(shape.end(), k.begin(), k.end());
        weight = std::make_shared<Tensor>(shape, true);
        init::kaiming_uniform_(weight, std::sqrt(5.0f));
        if (options.bias) {
            bias = std::make_shared<Tensor>(std::vector<int64_t>{out}, true);
            float bound = 1.0f / std::sqrt((float)init::fans(*weight).first);
            init::uniform_(bias, -bound, bound);
        }
    }
    ConvTransposeNd(int64_t in, int64_t out, int64_t kernel, int64_t stride = 1, int64_t padding = 0, int64_t output_padding = 0, int64_t groups = 1, bool use_bias = true)
        : ConvTransposeNd(in, out, std::vector<int64_t>{kernel}, make_options(stride, padding, output_padding, groups, use_bias)) {}

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override {
        return conv_transpose_nd(x, weight, bias, options.stride, options.padding, options.output_padding, options.dilation, options.groups);
    }
    void visit(ModuleVisitor& v) override {
        v.parameter("weight", weight);
        if (bias) v.parameter("bias", bias);
    }

    static ConvOptions make_options(int64_t stride, int64_t padding, int64_t output_padding, int64_t groups, bool use_bias) {
        ConvOptions o;
        o.stride = {stride};
        o.padding = {padding};
        o.output_padding = {output_padding};
        o.groups = groups;
        o.bias = use_bias;
        return o;
    }
};
using ConvTranspose1d = ConvTransposeNd<1>;
using ConvTranspose2d = ConvTransposeNd<2>;
using ConvTranspose3d = ConvTransposeNd<3>;

// ---------------------------------------------------------------------------------------------
// Pooling, upsampling and padding. The input's rank decides the number of spatial dimensions.
// ---------------------------------------------------------------------------------------------

class MaxPool : public Layer {
public:
    std::vector<int64_t> kernel, stride, padding, dilation;
    bool ceil_mode;
    explicit MaxPool(std::vector<int64_t> kernel, std::vector<int64_t> stride = {}, std::vector<int64_t> padding = {0}, std::vector<int64_t> dilation = {1}, bool ceil_mode = false)
        : kernel(std::move(kernel)), stride(std::move(stride)), padding(std::move(padding)), dilation(std::move(dilation)), ceil_mode(ceil_mode) {}
    explicit MaxPool(int64_t kernel) : MaxPool(std::vector<int64_t>{kernel}) {}
    MaxPool(int64_t kernel, int64_t stride, int64_t padding = 0) : MaxPool(std::vector<int64_t>{kernel}, std::vector<int64_t>{stride}, std::vector<int64_t>{padding}) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return max_pool_nd(x, kernel, stride, padding, dilation, ceil_mode).first; }
};
class AvgPool : public Layer {
public:
    std::vector<int64_t> kernel, stride, padding;
    explicit AvgPool(std::vector<int64_t> kernel, std::vector<int64_t> stride = {}, std::vector<int64_t> padding = {0})
        : kernel(std::move(kernel)), stride(std::move(stride)), padding(std::move(padding)) {}
    explicit AvgPool(int64_t kernel) : AvgPool(std::vector<int64_t>{kernel}) {}
    AvgPool(int64_t kernel, int64_t stride, int64_t padding = 0) : AvgPool(std::vector<int64_t>{kernel}, std::vector<int64_t>{stride}, std::vector<int64_t>{padding}) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return avg_pool_nd(x, kernel, stride, padding); }
};
using MaxPool1d = MaxPool;
using MaxPool2d = MaxPool;
using MaxPool3d = MaxPool;
using AvgPool1d = AvgPool;
using AvgPool2d = AvgPool;
using AvgPool3d = AvgPool;

class AdaptiveAvgPool : public Layer {
public:
    std::vector<int64_t> size;
    explicit AdaptiveAvgPool(std::vector<int64_t> size) : size(std::move(size)) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return adaptive_avg_pool_nd(x, size); }
};
class AdaptiveMaxPool : public Layer {
public:
    std::vector<int64_t> size;
    explicit AdaptiveMaxPool(std::vector<int64_t> size) : size(std::move(size)) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return adaptive_max_pool_nd(x, size); }
};
class Upsample : public Layer {
public:
    double scale;
    InterpolateMode mode;
    bool align_corners;
    explicit Upsample(double scale, InterpolateMode mode = InterpolateMode::Nearest, bool align_corners = false)
        : scale(scale), mode(mode), align_corners(align_corners) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return upsample(x, scale, mode, align_corners); }
};
class PixelShuffle : public Layer {
public:
    int64_t factor;
    explicit PixelShuffle(int64_t factor) : factor(factor) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return pixel_shuffle(x, factor); }
};
// Padding amounts in pad()'s order
class Pad : public Layer {
public:
    std::vector<int64_t> amounts;
    PadMode mode;
    float value;
    explicit Pad(std::vector<int64_t> amounts, PadMode mode = PadMode::Constant, float value = 0.0f) : amounts(std::move(amounts)), mode(mode), value(value) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override { return pad(x, amounts, mode, value); }
};

// ---------------------------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------------------------

// Normalizes over the batch and spatial dimensions of [N, C, ...] (or [N, C]). Training uses the batch's
// statistics and updates running ones (kept in buffers, saved in checkpoints); eval uses the running ones.
class BatchNorm : public Layer {
public:
    std::shared_ptr<Tensor> weight, bias, running_mean, running_var;
    float eps, momentum;

    explicit BatchNorm(int64_t features, float eps = 1e-5f, float momentum = 0.1f, bool affine = true) : eps(eps), momentum(momentum) {
        std::vector<int64_t> shape{features};
        if (affine) {
            weight = ones(shape, Device::CPU, DType::Float32, true);
            bias = zeros(shape, Device::CPU, DType::Float32, true);
        }
        running_mean = zeros(shape);
        running_var = ones(shape);
    }
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override {
        DG_PROFILE_OP("batch_norm");
        if (x->ndim() < 2 || x->shape[1] != running_mean->size()) throw std::invalid_argument("BatchNorm: input must be [N, " + std::to_string(running_mean->size()) + ", ...]");
        std::vector<int64_t> view((size_t)x->ndim(), 1);
        view[1] = x->shape[1];
        std::shared_ptr<Tensor> m, v;
        if (training_) {
            std::vector<int> dims{0};
            for (int d = 2; d < (int)x->ndim(); ++d) dims.push_back(d);
            int64_t n = x->size() / x->shape[1];
            if (n < 2) throw std::invalid_argument("BatchNorm: training needs more than one value per channel");
            m = mean(x, dims, true);
            v = mean(square(sub(x, m)), dims, true);
            NoGradGuard no_grad;
            auto blend = [&](const std::shared_ptr<Tensor>& running, const std::shared_ptr<Tensor>& batch) {
                ops::make_contiguous(*running, *add(mul_scalar(running, 1.0f - momentum), mul_scalar(reshape(batch->detach(), running->shape), momentum)));
            };
            blend(running_mean, m);
            blend(running_var, mul_scalar(v->detach(), (float)n / (float)(n - 1)));  // unbiased, as in PyTorch
        } else {
            m = reshape(running_mean, view);
            v = reshape(running_var, view);
        }
        auto y = mul(sub(x, m), rsqrt(add_scalar(v, eps)));
        return weight ? add(mul(y, reshape(weight, view)), reshape(bias, view)) : y;
    }
    void visit(ModuleVisitor& v) override {
        if (weight) {
            v.parameter("weight", weight);
            v.parameter("bias", bias);
        }
        v.buffer("running_mean", running_mean);
        v.buffer("running_var", running_var);
    }
};
using BatchNorm1d = BatchNorm;
using BatchNorm2d = BatchNorm;
using BatchNorm3d = BatchNorm;

// Normalizes groups of channels of [N, C, ...] within each sample
class GroupNorm : public Layer {
public:
    std::shared_ptr<Tensor> weight, bias;
    int64_t groups, channels;
    float eps;

    GroupNorm(int64_t groups, int64_t channels, float eps = 1e-5f, bool affine = true) : groups(groups), channels(channels), eps(eps) {
        if (groups < 1 || channels % groups != 0) throw std::invalid_argument("GroupNorm: groups must divide channels");
        if (affine) {
            weight = ones({channels}, Device::CPU, DType::Float32, true);
            bias = zeros({channels}, Device::CPU, DType::Float32, true);
        }
    }
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override {
        DG_PROFILE_OP("group_norm");
        if (x->ndim() < 2 || x->shape[1] != channels) throw std::invalid_argument("GroupNorm: input must be [N, " + std::to_string(channels) + ", ...]");
        auto g = reshape(x, {x->shape[0], groups, x->size() / (x->shape[0] * groups)});
        auto c = sub(g, mean(g, -1, true));
        auto y = reshape(mul(c, rsqrt(add_scalar(mean(square(c), -1, true), eps))), x->shape);
        if (!weight) return y;
        std::vector<int64_t> view((size_t)x->ndim(), 1);
        view[1] = channels;
        return add(mul(y, reshape(weight, view)), reshape(bias, view));
    }
    void visit(ModuleVisitor& v) override {
        if (!weight) return;
        v.parameter("weight", weight);
        v.parameter("bias", bias);
    }
};
// One group per channel
class InstanceNorm : public GroupNorm {
public:
    explicit InstanceNorm(int64_t channels, float eps = 1e-5f, bool affine = false) : GroupNorm(channels, channels, eps, affine) {}
};
using InstanceNorm1d = InstanceNorm;
using InstanceNorm2d = InstanceNorm;
using InstanceNorm3d = InstanceNorm;

// x / (k + alpha / size * sum of squares over neighbouring channels)^beta
class LocalResponseNorm : public Layer {
public:
    int64_t size;
    float alpha, beta, k;
    explicit LocalResponseNorm(int64_t size, float alpha = 1e-4f, float beta = 0.75f, float k = 1.0f) : size(size), alpha(alpha), beta(beta), k(k) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) override {
        int64_t C = x->shape[1];
        auto sq = square(x);
        TensorList sums;
        for (int64_t c = 0; c < C; ++c) {
            int64_t lo = std::max<int64_t>(0, c - size / 2), hi = std::min(C, c + (size - 1) / 2 + 1);
            sums.push_back(sum(narrow(sq, 1, lo, hi - lo), 1, true));
        }
        return div(x, pow(add_scalar(mul_scalar(cat(sums, 1), alpha / (float)size), k), beta));
    }
};

// x1 W x2 + b for batches of vectors
class Bilinear : public Module {
public:
    std::shared_ptr<Tensor> weight, bias;
    Bilinear(int64_t in1, int64_t in2, int64_t out) {
        weight = std::make_shared<Tensor>(std::vector<int64_t>{out, in1, in2}, true);
        bias = std::make_shared<Tensor>(std::vector<int64_t>{out}, true);
        float bound = 1.0f / std::sqrt((float)in1);
        init::uniform_(weight, -bound, bound);
        init::uniform_(bias, -bound, bound);
    }
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
        return add(einsum("...i,oij,...j->...o", {a, weight, b}), bias);
    }
    void visit(ModuleVisitor& v) override {
        v.parameter("weight", weight);
        v.parameter("bias", bias);
    }
};

// Sums, means or maxima of embedding rows per bag: ids [batch, bag size]
class EmbeddingBag : public Module {
public:
    enum class Mode { Sum, Mean, Max };
    std::shared_ptr<Tensor> weight;
    Mode mode;
    EmbeddingBag(int64_t num, int64_t dim, Mode mode = Mode::Mean) : mode(mode) {
        weight = std::make_shared<Tensor>(std::vector<int64_t>{num, dim}, true);
        init::normal_(weight);
    }
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& ids) {
        if (ids->ndim() != 2) throw std::invalid_argument("EmbeddingBag: ids must be [batch, bag size]");
        auto rows = reshape(index_select(weight, 0, reshape(ids, {ids->size()})), {ids->shape[0], ids->shape[1], weight->shape[1]});
        switch (mode) {
            case Mode::Sum: return sum(rows, 1, false);
            case Mode::Mean: return mean(rows, 1, false);
            default: return amax(rows, 1, false);
        }
    }
    void visit(ModuleVisitor& v) override { v.parameter("weight", weight); }
};

// ---------------------------------------------------------------------------------------------
// Reparameterizations: a weight computed from other parameters, as functions to call in forward
// ---------------------------------------------------------------------------------------------

// g * v / ||v||, the norm taken over every dimension but dim
inline std::shared_ptr<Tensor> weight_norm(const std::shared_ptr<Tensor>& v, const std::shared_ptr<Tensor>& g, int dim = 0) {
    std::vector<int> others;
    int keep = op_detail::wrap_dim(dim, v->ndim(), "weight_norm");
    for (int d = 0; d < (int)v->ndim(); ++d) if (d != keep) others.push_back(d);
    return mul(v, div(g, norm(v, 2.0f, others, true)));
}

// Divides a weight by its largest singular value, estimated by power iteration on a buffer that persists
// between calls (Miyato et al., 2018)
class SpectralNorm : public Module {
public:
    std::shared_ptr<Tensor> u;
    int iterations;
    float eps;
    explicit SpectralNorm(int64_t rows, int iterations = 1, float eps = 1e-12f) : iterations(iterations), eps(eps) {
        NoGradGuard no_grad;
        u = normalize(normal({rows}), 2.0f, 0, eps);
    }
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& weight) {
        auto m = reshape(weight, {weight->shape[0], weight->size() / weight->shape[0]});
        std::shared_ptr<Tensor> uc, vc;
        {
            NoGradGuard no_grad;
            auto w = m->detach();
            uc = u;
            int steps = training_ ? iterations : 1;
            for (int i = 0; i < steps; ++i) {
                vc = normalize(matmul(transpose(w, 0, 1), uc), 2.0f, 0, eps);
                auto next = normalize(matmul(w, vc), 2.0f, 0, eps);
                if (training_) uc = next;
            }
            if (training_) ops::make_contiguous(*u, *uc);
            uc = uc->clone();
        }
        auto sigma = dot(uc, matmul(m, vc));
        return div(weight, sigma);
    }
    void visit(ModuleVisitor& v) override { v.buffer("u", u); }
};
