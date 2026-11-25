#pragma once
// Dense linear algebra on the host, in double precision: the numerics behind linalg.h. Matrices are
// row-major.

#include <cstdint>
#include <vector>

namespace dense {

using Matrix = std::vector<double>;

// Lower Cholesky factor in place (the strict upper triangle is zeroed). False if not positive definite.
bool cholesky(Matrix& a, int64_t n);
// LU with partial pivoting in place: unit lower and upper triangle, pivots[i] = the row swapped with row i.
// Returns the permutation's sign, or 0 if singular (the factorization is still completed).
int lu(Matrix& a, int64_t n, std::vector<int64_t>& pivots);
// X = A^-1 B from lu's output; b is [n, m], overwritten
void lu_solve(const Matrix& a, int64_t n, const std::vector<int64_t>& pivots, Matrix& b, int64_t m);
// Solves T X = B for a triangular T [n, n]; b [n, m], overwritten
void triangular_solve(const Matrix& t, int64_t n, Matrix& b, int64_t m, bool upper, bool transpose, bool unit);
// Reduced QR by Householder reflections, R's diagonal made non-negative: q [m, k], r [k, n], k = min(m, n)
void qr(const Matrix& a, int64_t m, int64_t n, Matrix& q, Matrix& r);
// Eigenvalues (ascending) and orthonormal eigenvectors (columns of v) of a symmetric matrix: tridiagonal
// reduction and implicit QL
void eigh(const Matrix& a, int64_t n, std::vector<double>& w, Matrix& v);
// Thin SVD, singular values descending: u [m, k], s [k], vt [k, n]
void svd(const Matrix& a, int64_t m, int64_t n, Matrix& u, std::vector<double>& s, Matrix& vt);
// Eigenvalues of a general real matrix (Hessenberg reduction and shifted QR)
void eigvals(const Matrix& a, int64_t n, std::vector<double>& re, std::vector<double>& im);

// In-place complex DFT of n (re, im) pairs, unnormalized: radix-2 when n is a power of two, else Bluestein
void fft(std::vector<double>& x, int64_t n, bool inverse);

}  // namespace dense
