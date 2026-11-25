#include "parallel.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>

namespace {

thread_local bool t_in_region = false;

int default_threads() {
    if (const char* env = std::getenv("DUMMYGRAD_NUM_THREADS")) {
        int n = std::atoi(env);
        if (n > 0) return n;
    }
    int n = (int)std::thread::hardware_concurrency();
    return n > 0 ? n : 4;
}

std::atomic<int>& thread_count() {
    static std::atomic<int> n{default_threads()};
    return n;
}

// One parallel region: workers take pieces until none are left
struct Region {
    const std::function<void(int64_t, int64_t)>* body = nullptr;
    int64_t begin = 0, end = 0, pieces = 0;
    std::atomic<int64_t> next{0};
    std::atomic<int64_t> remaining{0};  // pieces not finished yet
    std::mutex error_mutex;
    std::exception_ptr error;
    std::mutex done_mutex;
    std::condition_variable done;

    void work() {
        bool previous = t_in_region;
        t_in_region = true;
        int64_t n = end - begin;
        for (int64_t p = next.fetch_add(1); p < pieces; p = next.fetch_add(1)) {
            int64_t b = begin + n * p / pieces, e = begin + n * (p + 1) / pieces;
            try {
                (*body)(b, e);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
            if (remaining.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lock(done_mutex);
                done.notify_all();
            }
        }
        t_in_region = previous;
    }
};

// Never destroyed: its threads are detached and may be waiting when the process exits
class Pool {
public:
    void run(const std::shared_ptr<Region>& region, int helpers) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            while ((int)started_ < helpers) {
                std::thread([this] { loop(); }).detach();
                ++started_;
            }
            current_ = region;
            wanted_ = helpers;
            ++generation_;
        }
        wake_.notify_all();
        region->work();
        // The caller ran out of pieces; wait for those other threads are still running
        for (int spin = 0; spin < 2000 && region->remaining.load() > 0; ++spin) std::this_thread::yield();
        {
            std::unique_lock<std::mutex> lock(region->done_mutex);
            region->done.wait(lock, [&] { return region->remaining.load() == 0; });
        }
        std::lock_guard<std::mutex> lock(mutex_);
        current_ = nullptr;
    }

private:
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            std::shared_ptr<Region> region;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return generation_ != seen && current_ && wanted_ > 0; });
                seen = generation_;
                --wanted_;
                region = current_;
            }
            region->work();
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::shared_ptr<Region> current_;
    uint64_t generation_ = 0;
    int wanted_ = 0;
    size_t started_ = 0;
};

Pool& pool() {
    static Pool* p = new Pool;
    return *p;
}

std::mutex& region_mutex() {
    static std::mutex* m = new std::mutex;
    return *m;
}

}  // namespace

int get_num_threads() { return thread_count().load(std::memory_order_relaxed); }

void set_num_threads(int n) { thread_count().store(std::max(1, n), std::memory_order_relaxed); }

namespace parallel_detail {

bool in_parallel_region() { return t_in_region; }

void run(int64_t begin, int64_t end, int64_t grain, const std::function<void(int64_t, int64_t)>& body) {
    // One region at a time: a second thread's kernel runs serially instead of oversubscribing the cores
    std::unique_lock<std::mutex> only(region_mutex(), std::try_to_lock);
    int64_t n = end - begin;
    int64_t most = (n + std::max<int64_t>(1, grain) - 1) / std::max<int64_t>(1, grain);
    int threads = (int)std::min<int64_t>(get_num_threads(), most);
    if (!only.owns_lock() || threads <= 1) {
        body(begin, end);
        return;
    }
    auto region = std::make_shared<Region>();
    region->body = &body;
    region->begin = begin;
    region->end = end;
    // More pieces than threads, so one slow core does not hold up the whole region
    region->pieces = std::min<int64_t>(most, (int64_t)threads * 4);
    region->remaining = region->pieces;
    pool().run(region, threads - 1);
    if (region->error) std::rethrow_exception(region->error);
}

}  // namespace parallel_detail
