#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// Op profiling. While enabled, every forward op and every backward node records its time and call
// count under its name: forward ops by function name ("matmul", "rms_norm", "add"), backward nodes
// by Node::name() ("MatmulBackward"). The GPU is synchronized at each op boundary so every time is
// exact, and backward runs one node at a time, so a profiled step is slower than a normal one: use
// it to find where time goes, not to time steps.
//
// Forward entries are self time. An op that calls other ops (relu densifying its input with
// contiguous, cross_entropy calling softmax) counts only its own work; each callee has its own
// entry. Ops a backward node runs internally are part of that node's time and get no entries.
//
//   reset_profile(); set_profiling(true);
//   ... a few training steps ...
//   set_profiling(false);
//   for (const auto& e : profile_report(ProfilePhase::Forward)) std::cout << e.name << " " << e.total_ms << "\n";
enum class ProfilePhase { Forward, Backward };

struct ProfileEntry {
    std::string name;
    ProfilePhase phase = ProfilePhase::Forward;
    int64_t calls = 0;
    double total_ms = 0.0;
};

namespace profiler_detail {
inline std::atomic<bool> enabled{false};
}  // namespace profiler_detail

void set_profiling(bool enabled);
inline bool profiling_enabled() { return profiler_detail::enabled.load(std::memory_order_relaxed); }
void reset_profile();
// Entries for both phases, or one, sorted by total time, largest first
std::vector<ProfileEntry> profile_report();
std::vector<ProfileEntry> profile_report(ProfilePhase phase);

// Times one forward op. name must outlive the scope (pass a string literal or a static name).
// Costs one relaxed atomic load when profiling is off. Put DG_PROFILE_OP("name") as the first
// statement of an op, before anything it times.
class ProfileScope {
public:
    explicit ProfileScope(const char* name) : name_(name) {
        if (profiling_enabled()) begin();
    }
    ~ProfileScope() {
        if (active_) end();
    }
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;

private:
    void begin();
    void end();
    const char* name_;
    bool active_ = false;
    ProfileScope* parent_ = nullptr;
    double start_ms_ = 0.0;
    double child_ms_ = 0.0;  // inclusive time of the ops this one called
};

#define DG_PROFILE_CONCAT_INNER(a, b) a##b
#define DG_PROFILE_CONCAT(a, b) DG_PROFILE_CONCAT_INNER(a, b)
#define DG_PROFILE_OP(name) ::ProfileScope DG_PROFILE_CONCAT(dg_profile_scope_, __LINE__)(name)

// Times one backward node (used by the autograd engine). Runs nodes one at a time, synchronizes
// around each, and keeps ops the node calls from recording forward entries.
class BackwardNodeTimer {
public:
    BackwardNodeTimer();
    ~BackwardNodeTimer();
    // Records the node's time; call once, after apply returns
    void finish(const std::string& name);
    BackwardNodeTimer(const BackwardNodeTimer&) = delete;
    BackwardNodeTimer& operator=(const BackwardNodeTimer&) = delete;

private:
    std::unique_lock<std::mutex> serial_;
    double start_ms_ = 0.0;
    bool previous_in_backward_;
};
