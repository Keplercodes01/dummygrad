#pragma once
#include "node.h"
#include "tensor.h"
#include "profiler.h"
#include <vector>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <string>

// Anomaly detection: when enabled, backward checks every gradient as it is produced and throws
// at the first NaN/Inf, naming the backward function that produced it. Costs a scan per
// gradient (and a device-to-host copy for GPU tensors), so enable it only while debugging.
// Global rather than thread-local because the checks run on the engine's worker threads.
void set_detect_anomaly(bool enabled);
bool detect_anomaly_enabled();

struct DetectAnomalyGuard {
    bool previous;
    DetectAnomalyGuard() : previous(detect_anomaly_enabled()) { set_detect_anomaly(true); }
    ~DetectAnomalyGuard() { set_detect_anomaly(previous); }
    DetectAnomalyGuard(const DetectAnomalyGuard&) = delete;
    DetectAnomalyGuard& operator=(const DetectAnomalyGuard&) = delete;
};

// Profiling (set_profiling, profile_report) for forward ops and backward nodes: see profiler.h

// Where backward runs its nodes. ThreadPool: the engine's worker threads, which can run
// independent branches of the graph at the same time. CallingThread: the thread that called
// backward(), one node at a time, most recently created first. On a GPU every kernel goes to the
// same stream, so worker threads add hand-off latency without adding parallelism: on a T4 the
// calling thread made a 0.4M-parameter fp16 step 2.1x faster and a 3.9M one 20% faster. Auto (the
// default) picks CallingThread. On the CPU the kernels themselves are multi-threaded, which is worth
// more than running branches side by side. Results are bit-identical in every mode: gradients that
// meet at a node are summed in a fixed order.
enum class BackwardExecution { Auto, ThreadPool, CallingThread };
void set_backward_execution(BackwardExecution mode);
BackwardExecution backward_execution();

class AutogradEngine {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;

    AutogradEngine();
    ~AutogradEngine();

public:
    static AutogradEngine& get() {
        static AutogradEngine instance;
        return instance;
    }
    
    // Disable copy/move
    AutogradEngine(const AutogradEngine&) = delete;
    AutogradEngine& operator=(const AutogradEngine&) = delete;

    struct Root {
        std::shared_ptr<Node> node;
        std::shared_ptr<Tensor> grad;  // the gradient of the final result with respect to this root
    };
    struct Options {
        bool retain_graph = false;
        bool create_graph = false;
        // When set, gradients arriving at these nodes are stored in captured (same order) instead of
        // being accumulated into .grad, and nodes that don't lead to one of them are not run
        std::vector<Node*> capture;
        std::vector<std::shared_ptr<Tensor>>* captured = nullptr;
    };
    void execute(const std::vector<Root>& roots, const Options& options);
};

// Differentiation as a function, as in torch.autograd.
namespace autograd {

// Gradients of the outputs with respect to the inputs, returned rather than accumulated into .grad.
// grad_outputs weight the outputs (a vector-Jacobian product); a missing one means 1 and needs a
// one-element output. create_graph=true records the gradient computation, so the result can be
// differentiated again:
//
//   auto g = autograd::grad({loss}, {x}, {}, true, true)[0];   // dloss/dx, differentiable
//   auto penalty = sum_all(mul(g, g));                         // gradient penalty
//   penalty->backward();
//
// An input the outputs don't depend on throws, or yields null with allow_unused. retain_graph
// defaults to create_graph.
std::vector<std::shared_ptr<Tensor>> grad(const std::vector<std::shared_ptr<Tensor>>& outputs,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs,
                                          const std::vector<std::shared_ptr<Tensor>>& grad_outputs = {},
                                          bool retain_graph = false, bool create_graph = false, bool allow_unused = false);

// Accumulates the gradients of the outputs into the .grad of every leaf they depend on
void backward(const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& grad_outputs = {}, bool retain_graph = false,
              bool create_graph = false);

}  // namespace autograd
