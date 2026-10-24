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

// Backward profiling: when enabled, the engine runs backward nodes one at a time and synchronizes
// the GPU around each, so every node's time is exact, and accumulates time and call counts per
// node name (Node::name()). This serializes backward, so enable it only to find where time goes.
struct ProfileEntry {
    std::string name;
    int64_t calls = 0;
    double total_ms = 0.0;
};
void set_profiling(bool enabled);
bool profiling_enabled();
void reset_profile();
// Entries sorted by total time, largest first
std::vector<ProfileEntry> profile_report();

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
