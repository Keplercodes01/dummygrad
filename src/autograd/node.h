#pragma once
#include <vector>
#include <memory>
#include <cstddef>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <string>

class Tensor;
struct Node;

// input_slot is the index into the *destination* node's incoming grads, not the
// position of the input in the producing op. Every node here has a single output,
// so it is always 0 — pass get_grad_edge(t).function with no slot argument.
struct Edge {
    std::shared_ptr<Node> function;
    size_t input_slot{0};
};

// Global creation counter: graphs are built in program order, so sequence numbers give the
// engine a deterministic order in which to sum gradients that meet at a node.
inline uint64_t next_node_sequence_nr() {
    static std::atomic<uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

struct Node : public std::enable_shared_from_this<Node> {
    std::vector<Edge> next_edges;
    std::mutex mutex;
    const uint64_t sequence_nr;
    Node() : sequence_nr(next_node_sequence_nr()) {}
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    virtual ~Node() = default;
    virtual std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) = 0;
    void add_next_edge(const std::shared_ptr<Node>& fn, size_t input_slot = 0);
    virtual void release_variables(){} 
    // Name used in error messages (e.g. anomaly detection); defaults to the C++ type name.
    virtual std::string name() const;
};

struct AccumulateGrad : public Node {
    std::weak_ptr<Tensor> variable;
    explicit AccumulateGrad(std::shared_ptr<Tensor> var) : variable(var) {}
    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override;
};
