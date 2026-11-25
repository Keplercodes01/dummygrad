#pragma once
// Shared by the test programs: running named tests, comparing values, making inputs.

#include "dummy_core.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

inline std::string g_filter;  // run only tests whose name contains this
inline int g_passed = 0;
inline std::vector<std::string> g_failures;
inline std::mt19937 g_rng(7);

inline void run(const std::string& name, const std::function<std::string()>& body) {
    if (!g_filter.empty() && name.find(g_filter) == std::string::npos) return;
    std::string err;
    try {
        err = body();
    } catch (const std::exception& e) {
        err = std::string("threw: ") + e.what();
    }
    if (err.empty()) {
        ++g_passed;
    } else {
        g_failures.push_back(name + ": " + err);
        std::cout << "[FAIL] " << name << ": " << err << std::endl;
    }
}

inline bool has_cuda() {
#ifdef USE_CUDA
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
    return false;
#endif
}

// Values uniform on [lo, hi], at least min_abs away from 0 and from each other's kinks
inline TensorPtr input(const std::vector<int64_t>& shape, float lo = -1.0f, float hi = 1.0f, float min_abs = 0.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::vector<float> v((size_t)n);
    for (auto& x : v) do { x = dist(g_rng); } while (std::fabs(x) < min_abs);
    return tensor(v, shape);
}
// Shuffled, evenly spaced values, so maxima and orderings survive a finite-difference step
inline TensorPtr distinct(const std::vector<int64_t>& shape, float gap = 0.15f, float offset = 0.05f) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::vector<float> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = gap * (float)(i - n / 2) + offset;
    std::shuffle(v.begin(), v.end(), g_rng);
    return tensor(v, shape);
}

inline std::string near(const std::vector<float>& a, const std::vector<float>& b, float atol, float rtol = 0.0f) {
    if (a.size() != b.size()) return "sizes " + std::to_string(a.size()) + " and " + std::to_string(b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i]) && std::isnan(b[i])) continue;
        if (a[i] == b[i]) continue;
        if (!(std::fabs(a[i] - b[i]) <= atol + rtol * std::max(std::fabs(a[i]), std::fabs(b[i])))) {
            std::ostringstream os;
            os << "element " << i << ": got " << a[i] << ", expected " << b[i];
            return os.str();
        }
    }
    return "";
}
inline std::string near(const TensorPtr& a, const std::vector<float>& b, float atol = 1e-5f, float rtol = 1e-5f) { return near(a->to_vector(), b, atol, rtol); }
// A template, so that a braced list is never taken for a tensor
template <typename T, typename = std::enable_if_t<std::is_same_v<T, TensorPtr>>>
inline std::string near(const TensorPtr& a, const T& b, float atol = 1e-5f, float rtol = 1e-5f) {
    if (a->shape != b->shape) return "shapes " + shape_string(a->shape) + " and " + shape_string(b->shape);
    return near(a->to_vector(), b->to_vector(), atol, rtol);
}
inline std::string scalar(const TensorPtr& a, float b, float atol = 1e-5f, float rtol = 1e-5f) { return near(std::vector<float>{a->item()}, {b}, atol, rtol); }
inline std::string shape_is(const TensorPtr& a, const std::vector<int64_t>& s) { return a->shape == s ? "" : "shape " + shape_string(a->shape) + ", expected " + shape_string(s); }
inline std::string expect_throw(const std::function<void()>& f, const std::string& what) {
    try {
        f();
    } catch (const std::exception&) {
        return "";
    }
    return what + " did not throw";
}
#define CHECK(expr)                             \
    do {                                        \
        std::string err_ = (expr);              \
        if (!err_.empty()) return #expr ": " + err_; \
    } while (0)

inline TensorPtr leaf_of(const TensorPtr& t) {
    if (t->requires_grad) return t;
    auto l = t->detach();
    l->requires_grad = true;
    return l;
}

inline int report() {
    std::cout << "\n" << g_passed << " passed, " << g_failures.size() << " failed\n";
    for (const auto& f : g_failures) std::cout << "  FAILED " << f << "\n";
    return g_failures.empty() ? 0 : 1;
}
