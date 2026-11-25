#pragma once
#include <cstdint>

namespace cpu {

// Row-major C = op_a(A) op_b(B) + beta C: op_a(A) is [m, k], op_b(B) [k, n]. BLAS when dummygrad was
// built with one, else a packed, vectorized, threaded implementation. A and B must not overlap C.
void gemm(bool trans_a, bool trans_b, int64_t m, int64_t n, int64_t k, const float* a, int64_t lda, const float* b,
          int64_t ldb, float beta, float* c, int64_t ldc);

}  // namespace cpu
