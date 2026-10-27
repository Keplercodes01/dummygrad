#include "profiler.h"
#include <algorithm>
#include <chrono>
#include <map>
#include <utility>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

std::mutex g_profile_mutex;   // guards g_profile
std::mutex g_profile_serial;  // runs one backward node at a time while profiling
std::map<std::pair<ProfilePhase, std::string>, ProfileEntry> g_profile;

// Innermost forward op running on this thread, and whether this thread is inside a backward node
thread_local ProfileScope* t_current_scope = nullptr;
thread_local bool t_in_backward_node = false;

void device_sync() {
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void record(ProfilePhase phase, const std::string& name, double ms) {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    auto& entry = g_profile[{phase, name}];
    entry.name = name;
    entry.phase = phase;
    entry.calls++;
    entry.total_ms += ms;
}

std::vector<ProfileEntry> sorted_entries(bool filter, ProfilePhase phase) {
    std::vector<ProfileEntry> out;
    {
        std::lock_guard<std::mutex> lock(g_profile_mutex);
        for (const auto& kv : g_profile) {
            if (!filter || kv.second.phase == phase) out.push_back(kv.second);
        }
    }
    std::sort(out.begin(), out.end(), [](const ProfileEntry& a, const ProfileEntry& b) { return a.total_ms > b.total_ms; });
    return out;
}

}  // namespace

void set_profiling(bool enabled) { profiler_detail::enabled = enabled; }

void reset_profile() {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    g_profile.clear();
}

std::vector<ProfileEntry> profile_report() { return sorted_entries(false, ProfilePhase::Forward); }
std::vector<ProfileEntry> profile_report(ProfilePhase phase) { return sorted_entries(true, phase); }

void ProfileScope::begin() {
    if (t_in_backward_node) return;  // part of the enclosing backward node's time
    active_ = true;
    parent_ = t_current_scope;
    t_current_scope = this;
    // Waits for work queued before this op (the caller's), so it is charged to the caller
    device_sync();
    start_ms_ = now_ms();
}

void ProfileScope::end() {
    device_sync();
    double inclusive = now_ms() - start_ms_;
    t_current_scope = parent_;
    if (parent_) parent_->child_ms_ += inclusive;
    record(ProfilePhase::Forward, name_, std::max(0.0, inclusive - child_ms_));
}

BackwardNodeTimer::BackwardNodeTimer() : serial_(g_profile_serial), previous_in_backward_(t_in_backward_node) {
    t_in_backward_node = true;
    device_sync();
    start_ms_ = now_ms();
}

BackwardNodeTimer::~BackwardNodeTimer() { t_in_backward_node = previous_in_backward_; }

void BackwardNodeTimer::finish(const std::string& name) {
    device_sync();
    record(ProfilePhase::Backward, name, now_ms() - start_ms_);
}
