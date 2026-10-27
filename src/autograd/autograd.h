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
