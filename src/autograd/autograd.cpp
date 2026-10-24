#include "autograd.h"
#include "ops.h"
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <typeinfo>
#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#include <cstdlib>
#endif
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

static std::atomic<bool> g_detect_anomaly{false};

void set_detect_anomaly(bool enabled) { g_detect_anomaly = enabled; }
bool detect_anomaly_enabled() { return g_detect_anomaly; }

static std::string node_name(const Node& node) {
    const char* mangled = typeid(node).name();
#if defined(__GNUC__) || defined(__clang__)
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    if (status == 0 && demangled) {
        std::string name(demangled);
        std::free(demangled);
        return name;
    }
#endif
    return mangled;
}

// Throws if any float gradient a node produced contains NaN or Inf.
static void check_finite_grads(const Node& node, const std::vector<std::shared_ptr<Tensor>>& grads) {
    for (size_t i = 0; i < grads.size(); ++i) {
        const auto& g = grads[i];
        if (!g || g->dtype != DType::Float32) continue;
        auto host = make_contiguous(g);
        if (host->device != Device::CPU) host = host->to(Device::CPU);
        const float* p = host->data_ptr<float>();
        for (int64_t k = 0; k < host->size(); ++k) {
            if (!std::isfinite(p[k])) {
                throw std::runtime_error("detect_anomaly: " + node_name(node) + " produced " +
                                         (std::isnan(p[k]) ? "NaN" : "Inf") + " in the gradient for its input " +
                                         std::to_string(i) + " (element " + std::to_string(k) + ")");
            }
        }
    }
}

AutogradEngine::AutogradEngine() {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    condition.wait(lock, [this]() { return stop || !tasks.empty(); });
                    if (stop && tasks.empty()) return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                try {
                    task();
                } catch (...) {
                    // Task handles exception reporting to execute()
                }
            }
        });
    }
}

AutogradEngine::~AutogradEngine() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
        if (worker.joinable()) worker.join();
    }
}

void AutogradEngine::execute(std::shared_ptr<Node> root, const std::vector<int64_t>& shape, Device device, DType dtype, bool retain_graph) {
    if (!root) return;

    // Precompute in-degrees (single threaded, iterative so deep graphs can't overflow the stack)
    std::unordered_map<Node*, int64_t> in_degree;
    in_degree[root.get()] = 0;
    std::vector<Node*> stack{root.get()};
    while (!stack.empty()) {
        Node* node = stack.back();
        stack.pop_back();
        for (const auto& edge : node->next_edges) {
            if (!edge.function) continue;
            Node* next = edge.function.get();
            if (!in_degree.count(next)) {
                in_degree[next] = 0;
                stack.push_back(next);
            }
            in_degree[next]++;
        }
    }

    // Gradients flowing into a node are collected, not summed on arrival: worker threads finish
    // in any order, and float addition is not associative. Once the node is ready they are summed
    // in a fixed order (producer creation order, then edge index), so identical runs produce
    // bitwise-identical gradients. Summing also happens outside the graph lock.
    struct Contribution {
        uint64_t producer;
        size_t edge;
        std::shared_ptr<Tensor> grad;
    };
    using Slot = std::vector<Contribution>;
    std::unordered_map<Node*, std::vector<Slot>> node_grads;
    std::mutex graph_mutex;

    auto seed_grad = std::make_shared<Tensor>(shape, device, dtype, false);
    seed_grad->fill_(1.0f);
    node_grads[root.get()] = {Slot{Contribution{0, 0, seed_grad}}};

    auto sum_slot = [](Slot& slot) -> std::shared_ptr<Tensor> {
        if (slot.empty()) return nullptr;
        if (slot.size() == 1) return slot[0].grad;
        std::sort(slot.begin(), slot.end(), [](const Contribution& a, const Contribution& b) {
            return a.producer != b.producer ? a.producer < b.producer : a.edge < b.edge;
        });
        // Producers may hand one tensor to several edges (AddBackward, views), so accumulate into
        // a fresh copy rather than into any contribution.
        std::shared_ptr<Tensor> total;
        for (auto& c : slot) tensor_add_inplace(total, c.grad);
        return total;
    };

    // Queued tasks reference this stack frame, so execute() must not return until every
    // scheduled task has finished -- including after a failure.
    int in_flight = 0;  // guarded by done_mutex
    std::mutex done_mutex;
    std::condition_variable done_cv;
    std::atomic<bool> failed{false};
    std::exception_ptr captured_exception = nullptr;  // guarded by done_mutex

    // Pool threads don't inherit the caller's current CUDA device. Without this, backward for
    // a replica on GPU N would launch its kernels on GPU 0.
#ifdef USE_CUDA
    int cuda_device = 0;
    cudaGetDevice(&cuda_device);
#endif

    std::function<void(std::shared_ptr<Node>)> push_task;

    auto run_node = [&](const std::shared_ptr<Node>& curr) {
        // All contributions have arrived once a node is scheduled, so its slots can be taken now
        std::vector<Slot> slots;
        {
            std::lock_guard<std::mutex> lock(graph_mutex);
            auto it = node_grads.find(curr.get());
            if (it != node_grads.end()) {
                slots = std::move(it->second);
                node_grads.erase(it);
            }
        }
        std::vector<std::shared_ptr<Tensor>> incoming;
        for (auto& slot : slots) incoming.push_back(sum_slot(slot));
        slots.clear();
        // Backward nodes index grads as dense row-major buffers, but view ops
        // (TransposeBackward) hand back strided views. Densify once here instead of in every node.
        for (auto& g : incoming) {
            if (g && !g->is_contiguous()) g = make_contiguous(g);
        }

        // MULTITHREADED EXECUTION: The expensive derivative math runs here!
        std::vector<std::shared_ptr<Tensor>> outgoing;
        if (!incoming.empty() && incoming[0]) {
            outgoing = curr->apply(incoming);
        }
        incoming.clear();
        if (detect_anomaly_enabled()) check_finite_grads(*curr, outgoing);

        for (size_t i = 0; i < curr->next_edges.size(); ++i) {
            if (failed) return;
            const auto& edge = curr->next_edges[i];
            if (!edge.function) continue;

            Node* next_node = edge.function.get();
            std::shared_ptr<Tensor> grad_to_pass = (i < outgoing.size()) ? outgoing[i] : nullptr;

            bool ready = false;
            {
                std::lock_guard<std::mutex> lock(graph_mutex);
                if (grad_to_pass) {
                    auto& next_slots = node_grads[next_node];
                    if (next_slots.size() <= edge.input_slot) next_slots.resize(edge.input_slot + 1);
                    next_slots[edge.input_slot].push_back(Contribution{curr->sequence_nr, i, grad_to_pass});
                }
                ready = (--in_degree[next_node] == 0);
            }
            if (ready) push_task(edge.function);
        }

        if (!retain_graph) {
            curr->next_edges.clear();
            curr->release_variables();
        }
    };

    push_task = [&](std::shared_ptr<Node> curr) {
        {
            std::lock_guard<std::mutex> lk(done_mutex);
            ++in_flight;
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.push([&, curr]() {
                try {
#ifdef USE_CUDA
                    cudaSetDevice(cuda_device);
#endif
                    if (!failed) run_node(curr);
                } catch (...) {
                    std::lock_guard<std::mutex> lk(done_mutex);
                    if (!captured_exception) captured_exception = std::current_exception();
                    failed = true;
                }
                // Decrement and notify under the lock: once in_flight hits 0 the waiting
                // thread may return and destroy done_mutex / done_cv.
                std::lock_guard<std::mutex> lk(done_mutex);
                if (--in_flight == 0) done_cv.notify_all();
            });
        }
        condition.notify_one();
    };

    push_task(root);

    {
        std::unique_lock<std::mutex> lock(done_mutex);
        done_cv.wait(lock, [&] { return in_flight == 0; });
    }

    if (captured_exception) {
        std::rethrow_exception(captured_exception);
    }
}
