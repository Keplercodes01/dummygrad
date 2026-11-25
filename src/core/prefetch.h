#pragma once
// Making items on background threads before they are needed, handed out in order.
//
//   Prefetcher<Batch> batches([&](int64_t i) { return load_batch(i); }, num_batches, /*workers=*/2);
//   Batch b;
//   while (batches.next(b)) train_on(b);

#include "grad_mode.h"
#include "parallel.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

template <typename T>
class Prefetcher {
public:
    // produce(i) makes item i. It runs on worker threads, without autograd, and must be safe to call from several at
    // once when there are several. Items i < count are made (count < 0: without end), at most `ahead` of them
    // before they are taken.
    Prefetcher(std::function<T(int64_t)> produce, int64_t count, int workers = 1, int ahead = 4)
        : produce_(std::move(produce)), count_(count), slots_((size_t)std::max(1, ahead)) {
        if (workers < 1) throw std::invalid_argument("Prefetcher: at least one worker is needed");
        for (int w = 0; w < workers; ++w) threads_.emplace_back([this] { work(); });
    }
    ~Prefetcher() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        room_.notify_all();
        for (auto& t : threads_) t.join();
    }
    Prefetcher(const Prefetcher&) = delete;
    Prefetcher& operator=(const Prefetcher&) = delete;

    // The next item; false after the last. An exception thrown while making an item is thrown here in its place.
    bool next(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (count_ >= 0 && taken_ >= count_) return false;
        Slot& s = slots_[(size_t)(taken_ % (int64_t)slots_.size())];
        ready_.wait(lock, [&] { return s.ready; });
        std::exception_ptr error = s.error;
        if (!error) out = std::move(s.item);
        s = Slot();
        ++taken_;
        lock.unlock();
        room_.notify_all();
        if (error) std::rethrow_exception(error);
        return true;
    }

private:
    struct Slot {
        T item{};
        std::exception_ptr error;
        bool ready = false;
    };

    void work() {
        NoGradGuard no_grad;
        SerialGuard serial;
        for (;;) {
            int64_t i;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                room_.wait(lock, [&] { return stop_ || (count_ >= 0 && claimed_ >= count_) || claimed_ < taken_ + (int64_t)slots_.size(); });
                if (stop_ || (count_ >= 0 && claimed_ >= count_)) return;
                i = claimed_++;
            }
            Slot made;
            made.ready = true;
            try {
                made.item = produce_(i);
            } catch (...) {
                made.error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                slots_[(size_t)(i % (int64_t)slots_.size())] = std::move(made);
            }
            ready_.notify_all();
        }
    }

    std::function<T(int64_t)> produce_;
    int64_t count_;
    std::vector<Slot> slots_;
    int64_t claimed_ = 0, taken_ = 0;
    bool stop_ = false;
    std::mutex mutex_;
    std::condition_variable room_, ready_;
    std::vector<std::thread> threads_;
};
