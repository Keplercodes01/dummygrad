#include "autograd.h"
#include "ops.h"
#include <unordered_map>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

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

    // A slot's first contribution is stored by reference. The producing node may have
    // handed that same tensor to other edges (AddBackward, AddScalarBackward, views),
    // so it is copied before any in-place accumulation touches it.
    struct GradSlot {
        std::shared_ptr<Tensor> grad;
        bool owned = false;
    };
    std::unordered_map<Node*, std::vector<GradSlot>> node_grads;
    std::mutex graph_mutex;

    auto seed_grad = std::make_shared<Tensor>(shape, device, dtype, false);
    seed_grad->fill_(1.0f);
    node_grads[root.get()] = {GradSlot{seed_grad, true}};

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
        std::vector<std::shared_ptr<Tensor>> incoming;
        {
            std::lock_guard<std::mutex> lock(graph_mutex);
            auto it = node_grads.find(curr.get());
            if (it != node_grads.end()) {
                for (auto& s : it->second) incoming.push_back(s.grad);
                node_grads.erase(it);
            }
        }
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
                    auto& slots = node_grads[next_node];
                    if (slots.size() <= edge.input_slot) slots.resize(edge.input_slot + 1);
                    auto& slot = slots[edge.input_slot];
                    if (!slot.grad) {
                        slot.grad = grad_to_pass;
                    } else {
                        if (!slot.owned) {
                            std::shared_ptr<Tensor> copy;
                            tensor_add_inplace(copy, slot.grad);  // null dst => deep copy
                            slot.grad = copy;
                            slot.owned = true;
                        }
                        tensor_add_inplace(slot.grad, grad_to_pass);
                    }
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
