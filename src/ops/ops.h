#pragma once
#include "tensor.h"
#include "functional.h"
#include <vector>

namespace ops {
    void fill(Tensor& tensor, float value);
    void add_inplace(Tensor& dst, const Tensor& src);
    void copy_from_vector(Tensor& tensor, const std::vector<float>& values);
    void make_contiguous(Tensor& dst, const Tensor& src);
}
