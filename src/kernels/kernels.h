#pragma once
// Device-agnostic kernels. Each one is written once (kernels_impl.inc, as functors over an output
// index) and compiled twice: for the host in kernels_host.cpp (threaded loops) and for the GPU in
// kernels_cuda.cu. Both backends therefore run the same arithmetic in the same order, and a kernel
// the CPU tests cover is the kernel the GPU runs.
//
//   DG_KERNEL(x->device, unary(kern::Unary::Exp, x->data_ptr<float>(), y->data_ptr<float>(), n, 0, 0));
//
// Pointers are float32 buffers on that device. Every kernel is deterministic: no atomics, and sums in
// an order fixed by the shapes.

#include "types.h"
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kern {

// X(Name): the elementwise functions of one input, with up to two constants p and q
#define DG_UNARY_LIST(X)                                                                              \
    X(Abs) X(Neg) X(Sign) X(Exp) X(Expm1) X(Log) X(Log2) X(Log10) X(Log1p) X(Sqrt) X(Rsqrt) X(Square) \
    X(Reciprocal) X(Sin) X(Cos) X(Tan) X(Asin) X(Acos) X(Atan) X(Sinh) X(Cosh) X(Tanh) X(Asinh)       \
    X(Acosh) X(Atanh) X(Erf) X(Erfc) X(Erfinv) X(Lgamma) X(Digamma) X(Trigamma) X(Floor) X(Ceil)      \
    X(Round) X(Trunc) X(IsNan) X(IsInf) X(IsFinite) X(LogicalNot) X(Clamp) X(PowScalar) X(ScalarPow)  \
    X(Sigmoid) X(LogSigmoid) X(Softplus) X(Relu) X(LeakyRelu) X(Elu) X(Gelu) X(GeluTanh) X(Silu)      \
    X(Mish) X(Hardswish) X(Hardsigmoid) X(Softsign) X(Softshrink) X(Hardshrink) X(Threshold)          \
    X(EqScalar) X(NeScalar) X(LtScalar) X(LeScalar) X(GtScalar) X(GeScalar) X(FmodScalar)             \
    X(RemainderScalar) X(InRange) X(Sinc) X(I0) X(I1) X(NanToNum)

#define DG_BINARY_LIST(X)                                                                        \
    X(Add) X(Sub) X(Mul) X(Div) X(Pow) X(Max) X(Min) X(Atan2) X(Fmod) X(Remainder) X(FloorDivide) \
    X(Hypot) X(Copysign) X(Eq) X(Ne) X(Lt) X(Le) X(Gt) X(Ge) X(And) X(Or) X(Xor)

#define DG_ENUM_ENTRY(name) name,
enum class Unary : int { DG_UNARY_LIST(DG_ENUM_ENTRY) };
enum class Binary : int { DG_BINARY_LIST(DG_ENUM_ENTRY) };
#undef DG_ENUM_ENTRY

enum class Reduce : int { Sum, Prod, Max, Min, SumSquares, SumAbs, CountNonzero };
enum class Cumulate : int { Sum, Prod, Max, Min, LogSumExp };

constexpr int kMaxDims = 8;

// Up to four strided operands walked over one shape. make_grid drops size-1 dimensions and merges
// neighbours every operand walks contiguously, so dense operands collapse to one long dimension.
struct Grid {
    int ndim = 1;
    int64_t n = 1;
    int64_t shape[kMaxDims] = {1};
    int64_t stride[4][kMaxDims] = {};
};

inline Grid make_grid(const std::vector<int64_t>& shape, const std::vector<std::vector<int64_t>>& strides) {
    Grid g;
    size_t ops = strides.size();
    if (ops > 4) throw std::invalid_argument("make_grid: at most 4 operands");
    std::vector<int64_t> dims;
    std::vector<std::vector<int64_t>> st(ops);
    g.n = 1;
    for (size_t d = 0; d < shape.size(); ++d) {
        g.n *= shape[d];
        if (shape[d] == 1) continue;
        bool merge = !dims.empty();
        for (size_t k = 0; k < ops && merge; ++k) merge = st[k].back() == strides[k][d] * shape[d];
        if (merge) {
            dims.back() *= shape[d];
            for (size_t k = 0; k < ops; ++k) st[k].back() = strides[k][d];
        } else {
            dims.push_back(shape[d]);
            for (size_t k = 0; k < ops; ++k) st[k].push_back(strides[k][d]);
        }
    }
    if (dims.empty()) {
        dims.push_back(1);
        for (auto& s : st) s.push_back(0);
    }
    if (dims.size() > (size_t)kMaxDims) throw std::invalid_argument("strided ops support at most 8 non-mergeable dimensions");
    g.ndim = (int)dims.size();
    for (size_t d = 0; d < dims.size(); ++d) {
        g.shape[d] = dims[d];
        for (size_t k = 0; k < ops; ++k) g.stride[k][d] = st[k][d];
    }
    return g;
}

// A gather along one dimension: out and index share `shape`; x is indexed with out's coordinates,
// except along dim, where the index tensor gives the coordinate
struct GatherPlan {
    int ndim = 1, dim = 0;
    int64_t n = 1;
    int64_t limit = 0;                 // x's size along dim
    int64_t shape[kMaxDims] = {1};     // index / output shape
    int64_t x_shape[kMaxDims] = {1};
    int64_t x_stride[kMaxDims] = {1};  // dense strides of x
    int64_t index_stride[kMaxDims] = {1};
};

// Sliding windows over up to three spatial dimensions (depth, height, width; unused leading ones have
// size 1). planes counts the (batch, channel) pairs.
struct Windows {
    int64_t planes = 1;
    int64_t in[3] = {1, 1, 1}, out[3] = {1, 1, 1};
    int64_t kernel[3] = {1, 1, 1}, stride[3] = {1, 1, 1}, pad[3] = {0, 0, 0}, dilation[3] = {1, 1, 1};
    bool count_include_pad = true;  // average pooling: divide by the window size, else by its valid part
    float divisor = 0.0f;           // average pooling: a fixed divisor when nonzero
    int64_t in_size() const { return in[0] * in[1] * in[2]; }
    int64_t out_size() const { return out[0] * out[1] * out[2]; }
    int64_t kernel_size() const { return kernel[0] * kernel[1] * kernel[2]; }
};

namespace host {
#include "kernels_decl.inc"
}
#ifdef USE_CUDA
namespace gpu {
#include "kernels_decl.inc"
}
#endif

}  // namespace kern

// CPU, MPS (unified memory) and TPU (host memory is authoritative) tensors run the host kernels
#ifdef USE_CUDA
#define DG_KERNEL(dev, call)                          \
    do {                                              \
        if ((dev) == Device::CUDA) kern::gpu::call;   \
        else kern::host::call;                        \
    } while (0)
#else
#define DG_KERNEL(dev, call)                                                                  \
    do {                                                                                      \
        if ((dev) == Device::CUDA) throw std::runtime_error("built without CUDA support");    \
        kern::host::call;                                                                     \
    } while (0)
#endif
