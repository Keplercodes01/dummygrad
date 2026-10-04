#include "autograd.h"
#include "ops.h"
#include <unordered_map>

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

    std::unordered_map<Node*, int64_t> in_degree;
    std::mutex graph_mutex; 

    // Precompute in-degrees (Single threaded)
    std::function<void(Node*)> compute_degrees = [&](Node* node) {
        if (!node) return;
        for (const auto& edge : node->next_edges) {
            if (edge.function) {
                if (!in_degree.count(edge.function.get())) {
                    in_degree[edge.function.get()] = 0;
                    compute_degrees(edge.function.get());
                }
                in_degree[edge.function.get()]++;
            }
        }
    };
    in_degree[root.get()] = 0;
    compute_degrees(root.get());

    std::unordered_map<Node*, std::vector<std::shared_ptr<Tensor>>> node_grads;
    auto seed_grad = std::make_shared<Tensor>(shape, device, dtype, false);
    seed_grad->fill_(1.0f);
    node_grads[root.get()] = {seed_grad};

    std::atomic<int> nodes_pending{(int)in_degree.size()}; 
    std::condition_variable done_cv;
    std::mutex done_mutex;

    std::exception_ptr captured_exception = nullptr;
    std::mutex exception_mutex;

    // We must pass push_task as a std::function so it can capture itself recursively
    std::function<void(std::shared_ptr<Node>)> push_task = [&](std::shared_ptr<Node> curr) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.push([curr, &node_grads, &in_degree, &graph_mutex, retain_graph, this,
                        &nodes_pending, &done_cv, &done_mutex, &push_task,
                        &captured_exception, &exception_mutex]() {
                
                struct PendingGuard {
                    std::atomic<int>& pending;
                    std::condition_variable& cv;
                    std::mutex& mtx;
                    ~PendingGuard() {
                        if (--pending == 0) {
                            std::lock_guard<std::mutex> lk(mtx);
                            cv.notify_all();
                        }
                    }
                } guard{nodes_pending, done_cv, done_mutex};

                try {
                    std::vector<std::shared_ptr<Tensor>> incoming;
                    {
                        std::lock_guard<std::mutex> lock(graph_mutex);
                        if (node_grads.count(curr.get())) {
                            incoming = node_grads[curr.get()];
                        }
                    }
                    
                    // MULTITHREADED EXECUTION: The expensive derivative math runs here!
                    std::vector<std::shared_ptr<Tensor>> outgoing;
                    if (!incoming.empty() && incoming[0]) {
                        outgoing = curr->apply(incoming);
                    }

                    // Immediately free input gradients to minimize peak memory
                    {
                        std::lock_guard<std::mutex> lock(graph_mutex);
                        node_grads.erase(curr.get());
                    }

                    for (size_t i = 0; i < curr->next_edges.size(); ++i) {
                        const auto& edge = curr->next_edges[i];
                        if (!edge.function) continue;

                        Node* next_node = edge.function.get();
                        std::shared_ptr<Tensor> grad_to_pass = (i < outgoing.size()) ? outgoing[i] : nullptr;
                        
                        bool ready = false;
                        {
                            std::lock_guard<std::mutex> lock(graph_mutex);
                            if (grad_to_pass) {
                                if (!node_grads.count(next_node)) {
                                    node_grads[next_node] = {grad_to_pass};
                                } else {
                                    tensor_add_inplace(node_grads[next_node][0], grad_to_pass);
                                }
                            }
                            in_degree[next_node]--;
                            if (in_degree[next_node] == 0) ready = true;
                        }
                        if (ready) {
                            push_task(edge.function);
                        }
                    }

                    if (!retain_graph) {
                        curr->next_edges.clear();
                        curr->release_variables();
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> elock(exception_mutex);
                    if (!captured_exception) {
                        captured_exception = std::current_exception();
                    }
                }
            });
        }
        condition.notify_one();
    };

    push_task(root);

    std::unique_lock<std::mutex> lock(done_mutex);
    done_cv.wait(lock, [&]{ return nodes_pending == 0; });

    if (captured_exception) {
        std::rethrow_exception(captured_exception);
    }
}
