#include "dense.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace dense {

bool cholesky(Matrix& a, int64_t n) {
    for (int64_t j = 0; j < n; ++j) {
        double d = a[j * n + j];
        for (int64_t k = 0; k < j; ++k) d -= a[j * n + k] * a[j * n + k];
        if (!(d > 0.0)) return false;
        d = std::sqrt(d);
        a[j * n + j] = d;
        for (int64_t i = j + 1; i < n; ++i) {
            double v = a[i * n + j];
            for (int64_t k = 0; k < j; ++k) v -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = v / d;
        }
        for (int64_t i = j + 1; i < n; ++i) a[j * n + i] = 0.0;
    }
    return true;
}

int lu(Matrix& a, int64_t n, std::vector<int64_t>& pivots) {
    int sign = 1;
    bool singular = false;
    pivots.resize((size_t)n);
    for (int64_t j = 0; j < n; ++j) {
        int64_t p = j;
        for (int64_t i = j + 1; i < n; ++i) if (std::fabs(a[i * n + j]) > std::fabs(a[p * n + j])) p = i;
        pivots[(size_t)j] = p;
        if (p != j) {
            for (int64_t k = 0; k < n; ++k) std::swap(a[p * n + k], a[j * n + k]);
            sign = -sign;
        }
        double d = a[j * n + j];
        if (d == 0.0) {
            singular = true;
            continue;
        }
        for (int64_t i = j + 1; i < n; ++i) {
            double f = a[i * n + j] /= d;
            if (f == 0.0) continue;
            for (int64_t k = j + 1; k < n; ++k) a[i * n + k] -= f * a[j * n + k];
        }
    }
    return singular ? 0 : sign;
}

void lu_solve(const Matrix& a, int64_t n, const std::vector<int64_t>& pivots, Matrix& b, int64_t m) {
    for (int64_t j = 0; j < n; ++j) {
        int64_t p = pivots[(size_t)j];
        if (p != j) for (int64_t k = 0; k < m; ++k) std::swap(b[p * m + k], b[j * m + k]);
    }
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < i; ++j)
            for (int64_t k = 0; k < m; ++k) b[i * m + k] -= a[i * n + j] * b[j * m + k];
    for (int64_t i = n - 1; i >= 0; --i) {
        for (int64_t j = i + 1; j < n; ++j)
            for (int64_t k = 0; k < m; ++k) b[i * m + k] -= a[i * n + j] * b[j * m + k];
        for (int64_t k = 0; k < m; ++k) b[i * m + k] /= a[i * n + i];
    }
}

void triangular_solve(const Matrix& t, int64_t n, Matrix& b, int64_t m, bool upper, bool transpose, bool unit) {
    auto at = [&](int64_t i, int64_t j) { return transpose ? t[j * n + i] : t[i * n + j]; };
    bool backward = upper != transpose;  // op(T) is upper triangular
    for (int64_t s = 0; s < n; ++s) {
        int64_t i = backward ? n - 1 - s : s;
        for (int64_t r = 0; r < s; ++r) {
            int64_t j = backward ? n - 1 - r : r;
            double v = at(i, j);
            if (v == 0.0) continue;
            for (int64_t k = 0; k < m; ++k) b[i * m + k] -= v * b[j * m + k];
        }
        if (!unit) for (int64_t k = 0; k < m; ++k) b[i * m + k] /= at(i, i);
    }
}

void qr(const Matrix& a_in, int64_t m, int64_t n, Matrix& q, Matrix& r) {
    int64_t k = std::min(m, n);
    Matrix a = a_in;
    std::vector<std::vector<double>> vs;
    for (int64_t j = 0; j < k; ++j) {
        std::vector<double> v((size_t)m, 0.0);
        double norm = 0.0;
        for (int64_t i = j; i < m; ++i) {
            v[(size_t)i] = a[i * n + j];
            norm += v[(size_t)i] * v[(size_t)i];
        }
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            v[(size_t)j] += v[(size_t)j] >= 0.0 ? norm : -norm;
            double vv = 0.0;
            for (int64_t i = j; i < m; ++i) vv += v[(size_t)i] * v[(size_t)i];
            for (int64_t c = j; c < n; ++c) {
                double dot = 0.0;
                for (int64_t i = j; i < m; ++i) dot += v[(size_t)i] * a[i * n + c];
                dot *= 2.0 / vv;
                for (int64_t i = j; i < m; ++i) a[i * n + c] -= dot * v[(size_t)i];
            }
            double scale = std::sqrt(2.0 / vv);
            for (auto& x : v) x *= scale;
        }
        vs.push_back(std::move(v));
    }
    r.assign((size_t)(k * n), 0.0);
    for (int64_t i = 0; i < k; ++i) for (int64_t j = i; j < n; ++j) r[i * n + j] = a[i * n + j];
    q.assign((size_t)(m * k), 0.0);
    for (int64_t i = 0; i < k; ++i) q[i * k + i] = 1.0;
    for (int64_t j = k - 1; j >= 0; --j) {
        const auto& v = vs[(size_t)j];
        for (int64_t c = 0; c < k; ++c) {
            double dot = 0.0;
            for (int64_t i = j; i < m; ++i) dot += v[(size_t)i] * q[i * k + c];
            for (int64_t i = j; i < m; ++i) q[i * k + c] -= dot * v[(size_t)i];
        }
    }
    for (int64_t i = 0; i < k; ++i) {
        if (r[i * n + i] >= 0.0) continue;
        for (int64_t j = 0; j < n; ++j) r[i * n + j] = -r[i * n + j];
        for (int64_t j = 0; j < m; ++j) q[j * k + i] = -q[j * k + i];
    }
}

// Householder tridiagonalization and implicit-shift QL (EISPACK tred2 / tql2)
void eigh(const Matrix& a, int64_t n, std::vector<double>& d, Matrix& v) {
    v = a;
    d.assign((size_t)n, 0.0);
    if (n == 0) return;
    // Use the symmetric part
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < i; ++j) v[i * n + j] = v[j * n + i] = 0.5 * (a[i * n + j] + a[j * n + i]);
    std::vector<double> e((size_t)n, 0.0);
    auto V = [&](int64_t i, int64_t j) -> double& { return v[(size_t)(i * n + j)]; };
    for (int64_t j = 0; j < n; ++j) d[(size_t)j] = V(n - 1, j);
    for (int64_t i = n - 1; i > 0; --i) {
        double scale = 0.0, h = 0.0;
        for (int64_t k = 0; k < i; ++k) scale += std::fabs(d[(size_t)k]);
        if (scale == 0.0) {
            e[(size_t)i] = d[(size_t)(i - 1)];
            for (int64_t j = 0; j < i; ++j) {
                d[(size_t)j] = V(i - 1, j);
                V(i, j) = 0.0;
                V(j, i) = 0.0;
            }
        } else {
            for (int64_t k = 0; k < i; ++k) {
                d[(size_t)k] /= scale;
                h += d[(size_t)k] * d[(size_t)k];
            }
            double f = d[(size_t)(i - 1)];
            double g = f > 0 ? -std::sqrt(h) : std::sqrt(h);
            e[(size_t)i] = scale * g;
            h -= f * g;
            d[(size_t)(i - 1)] = f - g;
            for (int64_t j = 0; j < i; ++j) e[(size_t)j] = 0.0;
            for (int64_t j = 0; j < i; ++j) {
                f = d[(size_t)j];
                V(j, i) = f;
                g = e[(size_t)j] + V(j, j) * f;
                for (int64_t k = j + 1; k <= i - 1; ++k) {
                    g += V(k, j) * d[(size_t)k];
                    e[(size_t)k] += V(k, j) * f;
                }
                e[(size_t)j] = g;
            }
            f = 0.0;
            for (int64_t j = 0; j < i; ++j) {
                e[(size_t)j] /= h;
                f += e[(size_t)j] * d[(size_t)j];
            }
            double hh = f / (h + h);
            for (int64_t j = 0; j < i; ++j) e[(size_t)j] -= hh * d[(size_t)j];
            for (int64_t j = 0; j < i; ++j) {
                f = d[(size_t)j];
                g = e[(size_t)j];
                for (int64_t k = j; k <= i - 1; ++k) V(k, j) -= f * e[(size_t)k] + g * d[(size_t)k];
                d[(size_t)j] = V(i - 1, j);
                V(i, j) = 0.0;
            }
        }
        d[(size_t)i] = h;
    }
    for (int64_t i = 0; i < n - 1; ++i) {
        V(n - 1, i) = V(i, i);
        V(i, i) = 1.0;
        double h = d[(size_t)(i + 1)];
        if (h != 0.0) {
            for (int64_t k = 0; k <= i; ++k) d[(size_t)k] = V(k, i + 1) / h;
            for (int64_t j = 0; j <= i; ++j) {
                double g = 0.0;
                for (int64_t k = 0; k <= i; ++k) g += V(k, i + 1) * V(k, j);
                for (int64_t k = 0; k <= i; ++k) V(k, j) -= g * d[(size_t)k];
            }
        }
        for (int64_t k = 0; k <= i; ++k) V(k, i + 1) = 0.0;
    }
    for (int64_t j = 0; j < n; ++j) {
        d[(size_t)j] = V(n - 1, j);
        V(n - 1, j) = 0.0;
    }
    V(n - 1, n - 1) = 1.0;
    e[0] = 0.0;

    for (int64_t i = 1; i < n; ++i) e[(size_t)(i - 1)] = e[(size_t)i];
    e[(size_t)(n - 1)] = 0.0;
    double f = 0.0, tst1 = 0.0;
    const double eps = 2.220446049250313e-16;
    for (int64_t l = 0; l < n; ++l) {
        tst1 = std::max(tst1, std::fabs(d[(size_t)l]) + std::fabs(e[(size_t)l]));
        int64_t m = l;
        while (m < n - 1 && std::fabs(e[(size_t)m]) > eps * tst1) ++m;
        if (m > l) {
            int iter = 0;
            do {
                if (++iter > 200) throw std::runtime_error("eigh: no convergence");
                double g = d[(size_t)l];
                double p = (d[(size_t)(l + 1)] - g) / (2.0 * e[(size_t)l]);
                double r = std::hypot(p, 1.0);
                if (p < 0) r = -r;
                d[(size_t)l] = e[(size_t)l] / (p + r);
                d[(size_t)(l + 1)] = e[(size_t)l] * (p + r);
                double dl1 = d[(size_t)(l + 1)];
                double h = g - d[(size_t)l];
                for (int64_t i = l + 2; i < n; ++i) d[(size_t)i] -= h;
                f += h;
                p = d[(size_t)m];
                double c = 1.0, c2 = c, c3 = c, el1 = e[(size_t)(l + 1)], s = 0.0, s2 = 0.0;
                for (int64_t i = m - 1; i >= l; --i) {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[(size_t)i];
                    h = c * p;
                    r = std::hypot(p, e[(size_t)i]);
                    e[(size_t)(i + 1)] = s * r;
                    s = e[(size_t)i] / r;
                    c = p / r;
                    p = c * d[(size_t)i] - s * g;
                    d[(size_t)(i + 1)] = h + s * (c * g + s * d[(size_t)i]);
                    for (int64_t k = 0; k < n; ++k) {
                        h = V(k, i + 1);
                        V(k, i + 1) = s * V(k, i) + c * h;
                        V(k, i) = c * V(k, i) - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[(size_t)l] / dl1;
                e[(size_t)l] = s * p;
                d[(size_t)l] = c * p;
            } while (std::fabs(e[(size_t)l]) > eps * tst1);
        }
        d[(size_t)l] += f;
        e[(size_t)l] = 0.0;
    }
    // Ascending order
    std::vector<int64_t> order((size_t)n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int64_t x, int64_t y) { return d[(size_t)x] < d[(size_t)y]; });
    std::vector<double> w((size_t)n);
    Matrix sorted((size_t)(n * n));
    for (int64_t j = 0; j < n; ++j) {
        w[(size_t)j] = d[(size_t)order[(size_t)j]];
        for (int64_t i = 0; i < n; ++i) sorted[(size_t)(i * n + j)] = V(i, order[(size_t)j]);
    }
    d = w;
    v = sorted;
}

// One-sided Jacobi: rotates pairs of columns until they are orthogonal. Accurate also for small
// singular values.
void svd(const Matrix& a_in, int64_t m, int64_t n, Matrix& u, std::vector<double>& s, Matrix& vt) {
    if (m < n) {
        Matrix at((size_t)(m * n));
        for (int64_t i = 0; i < m; ++i) for (int64_t j = 0; j < n; ++j) at[(size_t)(j * m + i)] = a_in[(size_t)(i * n + j)];
        Matrix u2, vt2;
        svd(at, n, m, u2, s, vt2);  // A^T = U2 S V2^T, so A = V2 S U2^T
        int64_t k = m;
        u.assign((size_t)(m * k), 0.0);
        vt.assign((size_t)(k * n), 0.0);
        for (int64_t i = 0; i < m; ++i) for (int64_t j = 0; j < k; ++j) u[(size_t)(i * k + j)] = vt2[(size_t)(j * m + i)];
        for (int64_t i = 0; i < k; ++i) for (int64_t j = 0; j < n; ++j) vt[(size_t)(i * n + j)] = u2[(size_t)(j * k + i)];
        return;
    }
    // Columns as rows, for contiguous access
    std::vector<double> c((size_t)(n * m)), v((size_t)(n * n), 0.0);
    for (int64_t i = 0; i < m; ++i) for (int64_t j = 0; j < n; ++j) c[(size_t)(j * m + i)] = a_in[(size_t)(i * n + j)];
    for (int64_t j = 0; j < n; ++j) v[(size_t)(j * n + j)] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        bool rotated = false;
        for (int64_t p = 0; p < n - 1; ++p) {
            for (int64_t q = p + 1; q < n; ++q) {
                double *cp = &c[(size_t)(p * m)], *cq = &c[(size_t)(q * m)];
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int64_t i = 0; i < m; ++i) {
                    alpha += cp[i] * cp[i];
                    beta += cq[i] * cq[i];
                    gamma += cp[i] * cq[i];
                }
                if (gamma == 0.0 || std::fabs(gamma) <= 1e-15 * std::sqrt(alpha * beta)) continue;
                rotated = true;
                double zeta = (beta - alpha) / (2.0 * gamma);
                double t = (zeta >= 0.0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                double cs = 1.0 / std::sqrt(1.0 + t * t), sn = cs * t;
                for (int64_t i = 0; i < m; ++i) {
                    double x = cp[i];
                    cp[i] = cs * x - sn * cq[i];
                    cq[i] = sn * x + cs * cq[i];
                }
                double *vp = &v[(size_t)(p * n)], *vq = &v[(size_t)(q * n)];
                for (int64_t i = 0; i < n; ++i) {
                    double x = vp[i];
                    vp[i] = cs * x - sn * vq[i];
                    vq[i] = sn * x + cs * vq[i];
                }
            }
        }
        if (!rotated) break;
    }
    std::vector<double> norms((size_t)n);
    for (int64_t j = 0; j < n; ++j) {
        double t = 0.0;
        for (int64_t i = 0; i < m; ++i) t += c[(size_t)(j * m + i)] * c[(size_t)(j * m + i)];
        norms[(size_t)j] = std::sqrt(t);
    }
    std::vector<int64_t> order((size_t)n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) { return norms[(size_t)x] > norms[(size_t)y]; });
    s.assign((size_t)n, 0.0);
    u.assign((size_t)(m * n), 0.0);
    vt.assign((size_t)(n * n), 0.0);
    std::vector<std::vector<double>> basis;  // U's columns so far, to complete it where s is 0
    for (int64_t k = 0; k < n; ++k) {
        int64_t j = order[(size_t)k];
        double sv = norms[(size_t)j];
        s[(size_t)k] = sv;
        for (int64_t i = 0; i < n; ++i) vt[(size_t)(k * n + i)] = v[(size_t)(j * n + i)];
        std::vector<double> col((size_t)m, 0.0);
        if (sv > 1e-300) {
            for (int64_t i = 0; i < m; ++i) col[(size_t)i] = c[(size_t)(j * m + i)] / sv;
        } else {
            for (int64_t t = 0; t < m; ++t) {
                std::vector<double> e((size_t)m, 0.0);
                e[(size_t)t] = 1.0;
                for (int pass = 0; pass < 2; ++pass) {
                    for (const auto& b : basis) {
                        double d = 0.0;
                        for (int64_t i = 0; i < m; ++i) d += b[(size_t)i] * e[(size_t)i];
                        for (int64_t i = 0; i < m; ++i) e[(size_t)i] -= d * b[(size_t)i];
                    }
                }
                double norm = 0.0;
                for (double x : e) norm += x * x;
                if (norm > 1e-6) {
                    norm = std::sqrt(norm);
                    for (int64_t i = 0; i < m; ++i) col[(size_t)i] = e[(size_t)i] / norm;
                    break;
                }
            }
        }
        for (int64_t i = 0; i < m; ++i) u[(size_t)(i * n + k)] = col[(size_t)i];
        basis.push_back(std::move(col));
    }
}

// Hessenberg reduction by elimination, then the Francis QR algorithm (EISPACK elmhes / hqr)
void eigvals(const Matrix& a_in, int64_t n, std::vector<double>& wr, std::vector<double>& wi) {
    Matrix h = a_in;
    wr.assign((size_t)n, 0.0);
    wi.assign((size_t)n, 0.0);
    auto A = [&](int64_t i, int64_t j) -> double& { return h[(size_t)(i * n + j)]; };
    for (int64_t m = 1; m < n - 1; ++m) {
        double x = 0.0;
        int64_t i = m;
        for (int64_t j = m; j < n; ++j) {
            if (std::fabs(A(j, m - 1)) > std::fabs(x)) {
                x = A(j, m - 1);
                i = j;
            }
        }
        if (i != m) {
            for (int64_t j = m - 1; j < n; ++j) std::swap(A(i, j), A(m, j));
            for (int64_t j = 0; j < n; ++j) std::swap(A(j, i), A(j, m));
        }
        if (x != 0.0) {
            for (i = m + 1; i < n; ++i) {
                double y = A(i, m - 1);
                if (y == 0.0) continue;
                y /= x;
                A(i, m - 1) = y;
                for (int64_t j = m; j < n; ++j) A(i, j) -= y * A(m, j);
                for (int64_t j = 0; j < n; ++j) A(j, m) += y * A(j, i);
            }
        }
    }
    for (int64_t i = 2; i < n; ++i) for (int64_t j = 0; j < i - 1; ++j) A(i, j) = 0.0;

    const double eps = 2.220446049250313e-16;
    double anorm = 0.0;
    for (int64_t i = 0; i < n; ++i) for (int64_t j = std::max<int64_t>(i - 1, 0); j < n; ++j) anorm += std::fabs(A(i, j));
    int64_t nn = n - 1;
    double t = 0.0, p = 0, q = 0, r = 0;
    while (nn >= 0) {
        int its = 0;
        int64_t l;
        do {
            for (l = nn; l >= 1; --l) {
                double s = std::fabs(A(l - 1, l - 1)) + std::fabs(A(l, l));
                if (s == 0.0) s = anorm;
                if (std::fabs(A(l, l - 1)) <= eps * s) {
                    A(l, l - 1) = 0.0;
                    break;
                }
            }
            double x = A(nn, nn);
            if (l == nn) {
                wr[(size_t)nn] = x + t;
                wi[(size_t)nn--] = 0.0;
            } else {
                double y = A(nn - 1, nn - 1), w = A(nn, nn - 1) * A(nn - 1, nn);
                if (l == nn - 1) {
                    p = 0.5 * (y - x);
                    q = p * p + w;
                    double z = std::sqrt(std::fabs(q));
                    x += t;
                    if (q >= 0.0) {
                        z = p + (p >= 0.0 ? z : -z);
                        wr[(size_t)(nn - 1)] = wr[(size_t)nn] = x + z;
                        if (z != 0.0) wr[(size_t)nn] = x - w / z;
                        wi[(size_t)(nn - 1)] = wi[(size_t)nn] = 0.0;
                    } else {
                        wr[(size_t)(nn - 1)] = wr[(size_t)nn] = x + p;
                        wi[(size_t)(nn - 1)] = z;
                        wi[(size_t)nn] = -z;
                    }
                    nn -= 2;
                } else {
                    if (its == 120) throw std::runtime_error("eigvals: no convergence");
                    if (its % 10 == 0 && its > 0) {
                        t += x;
                        for (int64_t i = 0; i <= nn; ++i) A(i, i) -= x;
                        double s = std::fabs(A(nn, nn - 1)) + std::fabs(A(nn - 1, nn - 2));
                        y = x = 0.75 * s;
                        w = -0.4375 * s * s;
                    }
                    ++its;
                    int64_t m;
                    double z = 0.0;
                    for (m = nn - 2; m >= l; --m) {
                        z = A(m, m);
                        r = x - z;
                        double s = y - z;
                        p = (r * s - w) / A(m + 1, m) + A(m, m + 1);
                        q = A(m + 1, m + 1) - z - r - s;
                        r = A(m + 2, m + 1);
                        s = std::fabs(p) + std::fabs(q) + std::fabs(r);
                        p /= s;
                        q /= s;
                        r /= s;
                        if (m == l) break;
                        double u = std::fabs(A(m, m - 1)) * (std::fabs(q) + std::fabs(r));
                        double v = std::fabs(p) * (std::fabs(A(m - 1, m - 1)) + std::fabs(z) + std::fabs(A(m + 1, m + 1)));
                        if (u <= eps * v) break;
                    }
                    for (int64_t i = m; i < nn - 1; ++i) {
                        A(i + 2, i) = 0.0;
                        if (i != m) A(i + 2, i - 1) = 0.0;
                    }
                    for (int64_t k = m; k < nn; ++k) {
                        if (k != m) {
                            p = A(k, k - 1);
                            q = A(k + 1, k - 1);
                            r = k + 1 != nn ? A(k + 2, k - 1) : 0.0;
                            x = std::fabs(p) + std::fabs(q) + std::fabs(r);
                            if (x != 0.0) {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        double s = std::sqrt(p * p + q * q + r * r);
                        if (p < 0.0) s = -s;
                        if (s == 0.0) continue;
                        if (k == m) {
                            if (l != m) A(k, k - 1) = -A(k, k - 1);
                        } else {
                            A(k, k - 1) = -s * x;
                        }
                        p += s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q /= p;
                        r /= p;
                        for (int64_t j = k; j <= nn; ++j) {
                            p = A(k, j) + q * A(k + 1, j);
                            if (k + 1 != nn) {
                                p += r * A(k + 2, j);
                                A(k + 2, j) -= p * z;
                            }
                            A(k + 1, j) -= p * y;
                            A(k, j) -= p * x;
                        }
                        int64_t mmin = nn < k + 3 ? nn : k + 3;
                        for (int64_t i = l; i <= mmin; ++i) {
                            p = x * A(i, k) + y * A(i, k + 1);
                            if (k + 1 != nn) {
                                p += z * A(i, k + 2);
                                A(i, k + 2) -= p * r;
                            }
                            A(i, k + 1) -= p * q;
                            A(i, k) -= p;
                        }
                    }
                }
            }
        } while (l < nn - 1 && nn >= 0);
    }
}

namespace {

void fft_radix2(std::vector<double>& x, int64_t n, bool inverse) {
    for (int64_t i = 1, j = 0; i < n; ++i) {
        int64_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(x[(size_t)(2 * i)], x[(size_t)(2 * j)]);
            std::swap(x[(size_t)(2 * i + 1)], x[(size_t)(2 * j + 1)]);
        }
    }
    const double pi = 3.14159265358979323846;
    for (int64_t len = 2; len <= n; len <<= 1) {
        double angle = (inverse ? 2.0 : -2.0) * pi / (double)len;
        int64_t half = len / 2;
        std::vector<double> tw((size_t)len);
        for (int64_t k = 0; k < half; ++k) {
            tw[(size_t)(2 * k)] = std::cos(angle * (double)k);
            tw[(size_t)(2 * k + 1)] = std::sin(angle * (double)k);
        }
        for (int64_t i = 0; i < n; i += len) {
            for (int64_t k = 0; k < half; ++k) {
                double *a = &x[(size_t)(2 * (i + k))], *b = &x[(size_t)(2 * (i + k + half))];
                double wr = tw[(size_t)(2 * k)], wi = tw[(size_t)(2 * k + 1)];
                double re = b[0] * wr - b[1] * wi, im = b[0] * wi + b[1] * wr;
                b[0] = a[0] - re;
                b[1] = a[1] - im;
                a[0] += re;
                a[1] += im;
            }
        }
    }
}

}  // namespace

void fft(std::vector<double>& x, int64_t n, bool inverse) {
    if (n <= 1) return;
    if ((n & (n - 1)) == 0) {
        fft_radix2(x, n, inverse);
        return;
    }
    // Bluestein: a DFT of any length as a convolution of a power-of-two length
    int64_t m = 1;
    while (m < 2 * n + 1) m <<= 1;
    const double pi = 3.14159265358979323846;
    std::vector<double> chirp((size_t)(2 * n)), a((size_t)(2 * m), 0.0), b((size_t)(2 * m), 0.0);
    for (int64_t k = 0; k < n; ++k) {
        int64_t sq = (int64_t)(((__int128)k * k) % (2 * n));
        double angle = (inverse ? 1.0 : -1.0) * pi * (double)sq / (double)n;
        chirp[(size_t)(2 * k)] = std::cos(angle);
        chirp[(size_t)(2 * k + 1)] = std::sin(angle);
        double cr = chirp[(size_t)(2 * k)], ci = chirp[(size_t)(2 * k + 1)];
        a[(size_t)(2 * k)] = x[(size_t)(2 * k)] * cr - x[(size_t)(2 * k + 1)] * ci;
        a[(size_t)(2 * k + 1)] = x[(size_t)(2 * k)] * ci + x[(size_t)(2 * k + 1)] * cr;
        b[(size_t)(2 * k)] = cr;
        b[(size_t)(2 * k + 1)] = -ci;
        if (k > 0) {
            b[(size_t)(2 * (m - k))] = cr;
            b[(size_t)(2 * (m - k) + 1)] = -ci;
        }
    }
    fft_radix2(a, m, false);
    fft_radix2(b, m, false);
    for (int64_t k = 0; k < m; ++k) {
        double re = a[(size_t)(2 * k)] * b[(size_t)(2 * k)] - a[(size_t)(2 * k + 1)] * b[(size_t)(2 * k + 1)];
        double im = a[(size_t)(2 * k)] * b[(size_t)(2 * k + 1)] + a[(size_t)(2 * k + 1)] * b[(size_t)(2 * k)];
        a[(size_t)(2 * k)] = re;
        a[(size_t)(2 * k + 1)] = im;
    }
    fft_radix2(a, m, true);
    for (int64_t k = 0; k < n; ++k) {
        double re = a[(size_t)(2 * k)] / (double)m, im = a[(size_t)(2 * k + 1)] / (double)m;
        double cr = chirp[(size_t)(2 * k)], ci = chirp[(size_t)(2 * k + 1)];
        x[(size_t)(2 * k)] = re * cr - im * ci;
        x[(size_t)(2 * k + 1)] = re * ci + im * cr;
    }
}

}  // namespace dense
