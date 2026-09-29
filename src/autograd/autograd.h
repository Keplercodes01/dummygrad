#pragma once
#include "node.h"
#include "tensor.h"
#include <vector>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

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

    void execute(std::shared_ptr<Node> root, const std::vector<int64_t>& shape, Device device, DType dtype, bool retain_graph);
};
