#include "autograd.h"
#include "functional.h"
#include "ops.h"
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

static std::atomic<bool> g_detect_anomaly{false};
static thread_local bool t_engine_worker = false;  // set on the engine's pool threads

void set_detect_anomaly(bool enabled) { g_detect_anomaly = enabled; }
bool detect_anomaly_enabled() { return g_detect_anomaly; }

static std::atomic<BackwardExecution> g_backward_execution{BackwardExecution::Auto};

void set_backward_execution(BackwardExecution mode) { g_backward_execution = mode; }
BackwardExecution backward_execution() { return g_backward_execution; }

// Runs node.apply, timing it when profiling is on.
static std::vector<std::shared_ptr<Tensor>> apply_node(Node& node, const std::vector<std::shared_ptr<Tensor>>& incoming,
                                                       bool create_graph) {
    // Gradients are only part of a graph themselves under create_graph
    GradModeGuard mode(create_graph);
    auto run = [&] { return create_graph ? node.apply_graph(incoming) : node.apply(incoming); };
    if (!profiling_enabled()) return run();
    BackwardNodeTimer timer;
    auto outgoing = run();
    timer.finish(node.name());
    return outgoing;
}

// Throws if any float gradient a node produced contains NaN or Inf.
static void check_finite_grads(const Node& node, const std::vector<std::shared_ptr<Tensor>>& grads) {
    for (size_t i = 0; i < grads.size(); ++i) {
        const auto& g = grads[i];
        if (!g || (g->dtype != DType::Float32 && g->dtype != DType::Float16 && g->dtype != DType::BFloat16)) continue;
        auto host = make_contiguous(g)->to(Device::CPU, DType::Float32);
        const float* p = host->data_ptr<float>();
        for (int64_t k = 0; k < host->size(); ++k) {
            if (!std::isfinite(p[k])) {
                throw std::runtime_error("detect_anomaly: " + node.name() + " produced " +
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
            t_engine_worker = true;
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

void AutogradEngine::execute(const std::vector<Root>& roots, const Options& options) {
    const bool create_graph = options.create_graph;
    const bool retain_graph = options.retain_graph || create_graph;
    GradModeGuard record(create_graph || grad_enabled());

    // How many gradients each node waits for. An ordinary backward runs every reachable node; with
    // captures, only those on the way to a captured node.
    std::unordered_map<Node*, int64_t> in_degree;
    std::unordered_map<Node*, size_t> capture_index;
    for (size_t i = 0; i < options.capture.size(); ++i) capture_index.emplace(options.capture[i], i);
    const bool capturing = options.captured != nullptr;
    std::unordered_set<Node*> needed, leads_on;  // when capturing: nodes to visit, and those with a needed node past them
    auto is_needed = [&](Node* n) { return !capturing || needed.count(n) > 0; };

    if (!capturing) {
        // Iterative, so deep graphs can't overflow the stack
        std::vector<Node*> stack;
        for (const auto& r : roots) {
            if (r.node && in_degree.emplace(r.node.get(), 0).second) stack.push_back(r.node.get());
        }
        while (!stack.empty()) {
            Node* node = stack.back();
            stack.pop_back();
            for (const auto& edge : node->next_edges) {
                Node* next = edge.function.get();
                if (!next) continue;
                auto entry = in_degree.emplace(next, 0);
                if (entry.second) stack.push_back(next);
                ++entry.first->second;
            }
        }
    } else {
        // Every reachable node, each after the nodes it feeds
        std::vector<Node*> order;
        std::unordered_set<Node*> seen;
        std::vector<std::pair<Node*, size_t>> stack;
        for (const auto& r : roots) {
            if (!r.node || !seen.insert(r.node.get()).second) continue;
            stack.emplace_back(r.node.get(), 0);
            while (!stack.empty()) {
                Node* node = stack.back().first;
                size_t i = stack.back().second;
                if (i == node->next_edges.size()) {
                    order.push_back(node);
                    stack.pop_back();
                    continue;
                }
                ++stack.back().second;
                Node* next = node->next_edges[i].function.get();
                if (next && seen.insert(next).second) stack.emplace_back(next, 0);
            }
        }
        for (Node* node : order) {
            bool on = false;
            for (const auto& edge : node->next_edges) on = on || (edge.function && needed.count(edge.function.get()));
            if (on) leads_on.insert(node);
            if (on || capture_index.count(node)) needed.insert(node);
        }
        for (Node* node : order) {
            if (!needed.count(node)) continue;
            in_degree.emplace(node, 0);
            if (!leads_on.count(node)) continue;  // it will not run, so it feeds nothing
            for (const auto& edge : node->next_edges) {
                if (edge.function && needed.count(edge.function.get())) in_degree[edge.function.get()]++;
            }
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

    for (size_t i = 0; i < roots.size(); ++i) {
        if (!roots[i].node || !is_needed(roots[i].node.get())) continue;
        auto& slots = node_grads[roots[i].node.get()];
        if (slots.empty()) slots.resize(1);
        auto g = roots[i].grad;
        if (!create_graph && g && g->requires_grad) g = saved_output(g);  // the seed's own history plays no part
        slots[0].push_back(Contribution{UINT64_MAX, i, std::move(g)});
    }

    auto sum_slot = [create_graph](Slot& slot) -> std::shared_ptr<Tensor> {
        if (slot.empty()) return nullptr;
        if (slot.size() == 1) return std::move(slot[0].grad);
        std::sort(slot.begin(), slot.end(), [](const Contribution& a, const Contribution& b) {
            return a.producer != b.producer ? a.producer < b.producer : a.edge < b.edge;
        });
        if (create_graph) {
            auto total = slot[0].grad;
            for (size_t k = 1; k < slot.size(); ++k) total = add(total, slot[k].grad);
            return total;
        }
        // Accumulate into one of the first two contributions when nothing else references it,
        // which saves copying it (a residual merge usually brings one fresh gradient). Float
        // addition is commutative, so a + b and b + a are bit-identical and the fixed order holds.
        // Producers may hand one tensor to several edges (AddBackward, views): grad_is_exclusive
        // rules those out, and without an exclusive one the sum goes into a fresh copy.
        size_t into = grad_is_exclusive(slot[0].grad) ? 0 : grad_is_exclusive(slot[1].grad) ? 1 : slot.size();
        std::shared_ptr<Tensor> total;
        if (into < slot.size()) total = std::move(slot[into].grad);
        for (size_t k = 0; k < slot.size(); ++k) {
            if (k != into) tensor_add_inplace(total, slot[k].grad);
        }
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
            if (g && !g->is_contiguous()) g = create_graph ? contiguous(g) : make_contiguous(g);
        }
        if (!curr->pre_hooks.empty() && !incoming.empty() && incoming[0]) {
            for (const auto& hook : curr->pre_hooks) {
                if (auto replaced = hook(incoming[0])) incoming[0] = make_contiguous(replaced);
            }
        }

        if (capturing) {
            auto it = capture_index.find(curr.get());
            if (it != capture_index.end() && !incoming.empty()) (*options.captured)[it->second] = incoming[0];
            if (!leads_on.count(curr.get())) return;
        }

        if (curr->released) {
            throw std::runtime_error(curr->name() + ": backward already ran through this part of the graph and freed it; "
                                     "pass retain_graph=true the first time to go through it again");
        }
        // The expensive derivative math runs here
        std::vector<std::shared_ptr<Tensor>> outgoing;
        if (!incoming.empty() && incoming[0]) {
            outgoing = apply_node(*curr, incoming, create_graph);
        }
        incoming.clear();
        if (detect_anomaly_enabled()) check_finite_grads(*curr, outgoing);

        for (size_t i = 0; i < curr->next_edges.size(); ++i) {
            if (failed) return;
            const auto& edge = curr->next_edges[i];
            if (!edge.function) continue;

            Node* next_node = edge.function.get();
            if (!is_needed(next_node)) continue;
            // Moved, not copied: the consumer may run before this loop ends, and AccumulateGrad
            // adopts a gradient only when it holds the sole reference.
            std::shared_ptr<Tensor> grad_to_pass = (i < outgoing.size()) ? std::move(outgoing[i]) : nullptr;

            bool ready = false;
            {
                std::lock_guard<std::mutex> lock(graph_mutex);
                if (grad_to_pass) {
                    auto& next_slots = node_grads[next_node];
                    if (next_slots.size() <= edge.input_slot) next_slots.resize(edge.input_slot + 1);
                    next_slots[edge.input_slot].push_back(Contribution{curr->sequence_nr, i, std::move(grad_to_pass)});
                }
                ready = (--in_degree[next_node] == 0);
            }
            if (ready) push_task(edge.function);
        }

        if (!retain_graph && !curr->next_edges.empty()) {
            curr->next_edges.clear();
            curr->release_variables();
            curr->released = true;
        }
    };

    // Roots nothing else leads to start; a root inside another root's graph waits for its gradients
    std::vector<std::shared_ptr<Node>> start;
    {
        std::unordered_set<Node*> pushed;
        for (const auto& r : roots) {
            if (r.node && is_needed(r.node.get()) && in_degree[r.node.get()] == 0 && pushed.insert(r.node.get()).second) {
                start.push_back(r.node);
            }
        }
    }

    BackwardExecution mode = g_backward_execution;
    // A backward started by a node (checkpointing) must not wait on the pool it is running on
    if (mode == BackwardExecution::Auto || create_graph || t_engine_worker) mode = BackwardExecution::CallingThread;
    if (mode == BackwardExecution::CallingThread) {
        // Newest node first, as PyTorch does: every node is created after the nodes it feeds
        // gradients to, so this follows the forward pass in reverse and frees saved tensors early.
        // An exception from a node propagates straight out; nothing else is in flight.
        auto older = [](const std::shared_ptr<Node>& x, const std::shared_ptr<Node>& y) {
            return x->sequence_nr < y->sequence_nr;
        };
        std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, decltype(older)> ready(older);
        push_task = [&](std::shared_ptr<Node> curr) { ready.push(std::move(curr)); };
        for (auto& node : start) ready.push(std::move(node));
        start.clear();
        while (!ready.empty()) {
            auto curr = ready.top();
            ready.pop();
            run_node(curr);
        }
        return;
    }

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

    for (auto& node : start) push_task(node);
    start.clear();

    {
        std::unique_lock<std::mutex> lock(done_mutex);
        done_cv.wait(lock, [&] { return in_flight == 0; });
    }

    if (captured_exception) {
        std::rethrow_exception(captured_exception);
    }
}

namespace autograd {

namespace {

// The seed gradient for each output, checked, and the node it enters the graph at
std::vector<AutogradEngine::Root> make_roots(const char* who, const std::vector<std::shared_ptr<Tensor>>& outputs,
                                             const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
    if (!grad_outputs.empty() && grad_outputs.size() != outputs.size()) {
        throw std::invalid_argument(std::string(who) + ": grad_outputs must be empty or match outputs");
    }
    std::vector<AutogradEngine::Root> roots;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& out = outputs[i];
        if (!out) throw std::invalid_argument(std::string(who) + ": null output");
        if (!out->requires_grad) throw std::runtime_error(std::string(who) + ": an output does not require grad (nothing to differentiate)");
        auto g = i < grad_outputs.size() ? grad_outputs[i] : nullptr;
        if (!g) {
            if (out->size() != 1) throw std::runtime_error(std::string(who) + ": an output with more than one element needs a grad_output");
            g = std::make_shared<Tensor>(out->shape, out->device, out->dtype, false);
            g->fill_(1.0f);
        } else if (g->shape != out->shape) {
            throw std::invalid_argument(std::string(who) + ": grad_output " + shape_string(g->shape) + " does not match output " +
                                        shape_string(out->shape));
        }
        roots.push_back({get_grad_edge(out).function, g});
    }
    return roots;
}

}  // namespace

std::vector<std::shared_ptr<Tensor>> grad(const std::vector<std::shared_ptr<Tensor>>& outputs,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs,
                                          const std::vector<std::shared_ptr<Tensor>>& grad_outputs, bool retain_graph,
                                          bool create_graph, bool allow_unused) {
    GradModeGuard enable(true);  // graph edges are looked up with recording on, whatever the caller's mode
    auto roots = make_roots("autograd::grad", outputs, grad_outputs);
    AutogradEngine::Options opt;
    opt.retain_graph = retain_graph || create_graph;
    opt.create_graph = create_graph;
    std::unordered_map<Node*, size_t> slot_of;  // an input listed twice shares one capture
    std::vector<std::shared_ptr<Node>> keep;
    std::vector<size_t> slot(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i] || !inputs[i]->requires_grad) throw std::runtime_error("autograd::grad: an input does not require grad");
        auto node = get_grad_edge(inputs[i]).function;
        auto it = slot_of.find(node.get());
        if (it == slot_of.end()) {
            it = slot_of.emplace(node.get(), opt.capture.size()).first;
            opt.capture.push_back(node.get());
            keep.push_back(node);
        }
        slot[i] = it->second;
    }
    std::vector<std::shared_ptr<Tensor>> captured(opt.capture.size());
    opt.captured = &captured;
    AutogradEngine::get().execute(roots, opt);
    std::vector<std::shared_ptr<Tensor>> result(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        result[i] = captured[slot[i]];
        if (!result[i] && !allow_unused) {
            throw std::runtime_error("autograd::grad: an input was not used to compute the outputs (pass allow_unused=true to get null)");
        }
    }
    return result;
}

void backward(const std::vector<std::shared_ptr<Tensor>>& outputs, const std::vector<std::shared_ptr<Tensor>>& grad_outputs,
              bool retain_graph, bool create_graph) {
    GradModeGuard enable(true);
    auto roots = make_roots("autograd::backward", outputs, grad_outputs);
    AutogradEngine::Options opt;
    opt.retain_graph = retain_graph;
    opt.create_graph = create_graph;
    AutogradEngine::get().execute(roots, opt);
}

}  // namespace autograd
