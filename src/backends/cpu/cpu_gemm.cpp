#include "cpu_gemm.h"
#include "parallel.h"

#include <algorithm>
#include <cstring>
#include <vector>

#if defined(USE_BLAS) || defined(USE_OPENBLAS) || defined(USE_MKL)
extern "C" void cblas_sgemm(int layout, int trans_a, int trans_b, const int m, const int n, const int k, const float alpha,
                            const float* a, const int lda, const float* b, const int ldb, const float beta, float* c,
                            const int ldc);
#define DG_HAS_BLAS 1
#else
#define DG_HAS_BLAS 0
#endif

namespace cpu {

#if DG_HAS_BLAS

void gemm(bool trans_a, bool trans_b, int64_t m, int64_t n, int64_t k, const float* a, int64_t lda, const float* b,
          int64_t ldb, float beta, float* c, int64_t ldc) {
    if (m == 0 || n == 0) return;
    const int kRowMajor = 101, kNoTrans = 111, kTrans = 112;
    cblas_sgemm(kRowMajor, trans_a ? kTrans : kNoTrans, trans_b ? kTrans : kNoTrans, (int)m, (int)n, (int)k, 1.0f, a,
                (int)std::max<int64_t>(1, lda), b, (int)std::max<int64_t>(1, ldb), beta, c, (int)std::max<int64_t>(1, ldc));
}

#else

namespace {

// Vector registers of the target: NR columns of C are computed at once, in NV registers
#if defined(__AVX512F__)
constexpr int kVectorBytes = 64, kRegs = 2, kMR = 12;
#elif defined(__AVX__)
constexpr int kVectorBytes = 32, kRegs = 2, kMR = 6;
#elif defined(__ARM_NEON)
constexpr int kVectorBytes = 16, kRegs = 4, kMR = 6;
#else
constexpr int kVectorBytes = 16, kRegs = 2, kMR = 4;
#endif
constexpr int kLanes = kVectorBytes / 4;
constexpr int kNR = kLanes * kRegs;
constexpr int64_t kKC = 256, kMC = 20 * kMR, kNC = 128 * kNR;

typedef float Vec __attribute__((vector_size(kVectorBytes)));

inline Vec load(const float* p) {
    Vec v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
inline void store(float* p, Vec v) { std::memcpy(p, &v, sizeof(v)); }

// C[kMR, kNR] += packed A panel times packed B panel
void micro_kernel(int64_t kc, const float* a, const float* b, float* c, int64_t ldc) {
    Vec acc[kMR][kRegs];
    for (int i = 0; i < kMR; ++i)
        for (int r = 0; r < kRegs; ++r) acc[i][r] = Vec{};
    for (int64_t p = 0; p < kc; ++p) {
        Vec bv[kRegs];
        for (int r = 0; r < kRegs; ++r) bv[r] = load(b + r * kLanes);
        for (int i = 0; i < kMR; ++i) {
            Vec av = Vec{} + a[i];
            for (int r = 0; r < kRegs; ++r) acc[i][r] += av * bv[r];
        }
        a += kMR;
        b += kNR;
    }
    for (int i = 0; i < kMR; ++i)
        for (int r = 0; r < kRegs; ++r) store(c + i * ldc + r * kLanes, load(c + i * ldc + r * kLanes) + acc[i][r]);
}

void pack_a(bool trans, const float* a, int64_t lda, int64_t i0, int64_t mc, int64_t p0, int64_t kc, float* out) {
    for (int64_t i = 0; i < mc; i += kMR) {
        int64_t rows = std::min<int64_t>(kMR, mc - i);
        if (rows < kMR) std::fill(out, out + kMR * kc, 0.0f);
        if (trans) {
            // op(A)'s rows are A's columns: each packed row of kMR is contiguous in memory
            const float* src = a + p0 * lda + i0 + i;
            if (rows == kMR) for (int64_t p = 0; p < kc; ++p) std::memcpy(out + p * kMR, src + p * lda, kMR * sizeof(float));
            else for (int64_t p = 0; p < kc; ++p) std::memcpy(out + p * kMR, src + p * lda, (size_t)rows * sizeof(float));
        } else {
            for (int64_t r = 0; r < rows; ++r) {
                const float* src = a + (i0 + i + r) * lda + p0;
                for (int64_t p = 0; p < kc; ++p) out[p * kMR + r] = src[p];
            }
        }
        out += kMR * kc;
    }
}

void pack_b_panel(bool trans, const float* b, int64_t ldb, int64_t p0, int64_t kc, int64_t j0, int64_t cols, float* out) {
    for (int64_t p = 0; p < kc; ++p) {
        if (!trans) std::memcpy(out, b + (p0 + p) * ldb + j0, (size_t)cols * sizeof(float));
        else for (int64_t r = 0; r < cols; ++r) out[r] = b[(j0 + r) * ldb + p0 + p];
        for (int64_t r = cols; r < kNR; ++r) out[r] = 0.0f;
        out += kNR;
    }
}

}  // namespace

void gemm(bool trans_a, bool trans_b, int64_t m, int64_t n, int64_t k, const float* a, int64_t lda, const float* b,
          int64_t ldb, float beta, float* c, int64_t ldc) {
    for (int64_t i = 0; i < m; ++i) {
        if (beta == 0.0f) std::fill(c + i * ldc, c + i * ldc + n, 0.0f);
        else if (beta != 1.0f) for (int64_t j = 0; j < n; ++j) c[i * ldc + j] *= beta;
    }
    if (m == 0 || n == 0 || k == 0) return;
    std::vector<float> packed_b((size_t)(std::min(n + kNR, kNC) * std::min(k, kKC)));
    for (int64_t jc = 0; jc < n; jc += kNC - kNR) {
        int64_t nc = std::min(kNC - kNR, n - jc);
        int64_t panels = (nc + kNR - 1) / kNR;
        for (int64_t pc = 0; pc < k; pc += kKC) {
            int64_t kc = std::min(kKC, k - pc);
            parallel_for(0, panels, std::max<int64_t>(1, 65536 / (kc * kNR)), [&](int64_t begin, int64_t end) {
                for (int64_t p = begin; p < end; ++p)
                    pack_b_panel(trans_b, b, ldb, pc, kc, jc + p * kNR, std::min<int64_t>(kNR, nc - p * kNR), packed_b.data() + p * kNR * kc);
            });
            int64_t blocks = (m + kMC - 1) / kMC;
            int64_t grain = std::max<int64_t>(1, 200000 / (kMC * nc * kc / 8 + 1));
            parallel_for(0, blocks, grain, [&](int64_t begin, int64_t end) {
                std::vector<float> packed_a((size_t)(kMC * kc));
                float tail[kMR * kNR];
                for (int64_t blk = begin; blk < end; ++blk) {
                    int64_t ic = blk * kMC, mc = std::min(kMC, m - ic);
                    pack_a(trans_a, a, lda, ic, mc, pc, kc, packed_a.data());
                    for (int64_t p = 0; p < panels; ++p) {
                        int64_t j = p * kNR, cols = std::min<int64_t>(kNR, nc - j);
                        for (int64_t i = 0; i < mc; i += kMR) {
                            int64_t rows = std::min<int64_t>(kMR, mc - i);
                            const float* ap = packed_a.data() + i * kc;
                            const float* bp = packed_b.data() + p * kNR * kc;
                            float* cp = c + (ic + i) * ldc + jc + j;
                            if (rows == kMR && cols == kNR) {
                                micro_kernel(kc, ap, bp, cp, ldc);
                            } else {
                                std::fill(tail, tail + kMR * kNR, 0.0f);
                                micro_kernel(kc, ap, bp, tail, kNR);
                                for (int64_t r = 0; r < rows; ++r)
                                    for (int64_t s = 0; s < cols; ++s) cp[r * ldc + s] += tail[r * kNR + s];
                            }
                        }
                    }
                }
            });
        }
    }
}

#endif

}  // namespace cpu
