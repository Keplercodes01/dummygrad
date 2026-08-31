#include "utils.h"
#include <cassert>

std::vector<int64_t> make_strides(const std::vector<int64_t>& shape) {
    int64_t ndim = (int64_t)shape.size();
    assert(ndim > 0 && "make_strides: shape cannot be empty");
    std::vector<int64_t> st(ndim);
    st[ndim - 1] = 1; 
    for (int64_t i = ndim - 2; i >= 0; i--) 
        st[i] = st[i + 1] * shape[i + 1];
    return st;
}

std::vector<int64_t> unravel(int64_t flat, const std::vector<int64_t>& shape) {
    int64_t ndim = (int64_t)shape.size();
    if (ndim == 0) return {};
    assert(flat >= 0 && "unravel: flat index cannot be negative");
    std::vector<int64_t> idx(ndim);
    for (int64_t d = ndim - 1; d >= 0; d--) {
        idx[d] = flat % shape[d];
        flat /= shape[d];
    }
    return idx;
}
