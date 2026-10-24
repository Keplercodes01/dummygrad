#pragma once

// Gradient mode: whether ops record the autograd graph. Thread-local, like PyTorch's, so
// disabling it on one thread (e.g. an eval loop) does not affect others.
inline thread_local bool g_grad_enabled = true;

inline bool grad_enabled() { return g_grad_enabled; }
inline void set_grad_enabled(bool enabled) { g_grad_enabled = enabled; }

// RAII: no graph is recorded while a NoGradGuard is alive. Outputs don't require grad and
// keep no references to their inputs, so evaluation and inference use less memory.
//
//   { NoGradGuard no_grad; auto preds = model.forward(x); }
struct NoGradGuard {
    bool previous;
    NoGradGuard() : previous(g_grad_enabled) { g_grad_enabled = false; }
    ~NoGradGuard() { g_grad_enabled = previous; }
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
};
