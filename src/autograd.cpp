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
                task();
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

    // We must pass push_task as a std::function so it can capture itself recursively
    std::function<void(std::shared_ptr<Node>)> push_task = [&](std::shared_ptr<Node> curr) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.push([curr, &node_grads, &in_degree, &graph_mutex, retain_graph, this, &nodes_pending, &done_cv, &push_task]() {
                
                std::vector<std::shared_ptr<Tensor>> incoming;
                {
                    std::lock_guard<std::mutex> lock(graph_mutex);
                    incoming = node_grads[curr.get()];
                }
                
                // MULTITHREADED EXECUTION: The expensive derivative math runs here!
                std::vector<std::shared_ptr<Tensor>> outgoing = curr->apply(incoming);

                for (size_t i = 0; i < curr->next_edges.size(); ++i) {
                    const auto& edge = curr->next_edges[i];
                    if (!edge.function) continue;

                    Node* next_node = edge.function.get();
                    std::shared_ptr<Tensor> grad_to_pass = (i < outgoing.size()) ? outgoing[i] : nullptr;
                    
                    if (grad_to_pass) {
                        std::lock_guard<std::mutex> lock(next_node->mutex); // PREVENT RACE CONDITION
                        if (!node_grads.count(next_node)) {
                            node_grads[next_node] = {grad_to_pass};
                        } else {
                            tensor_add_inplace(node_grads[next_node][0], grad_to_pass);
                        }
                    }

                    bool ready = false;
                    {
                        std::lock_guard<std::mutex> lock(graph_mutex);
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

                nodes_pending--;
                if (nodes_pending == 0) {
                    done_cv.notify_all();
                }
            });
        }
        condition.notify_one();
    };

    push_task(root);

    std::unique_lock<std::mutex> lock(done_mutex);
    done_cv.wait(lock, [&]{ return nodes_pending == 0; });
}
