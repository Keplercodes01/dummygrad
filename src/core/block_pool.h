#pragma once
// Carves blocks out of large segments of memory: segments are split as requests need, and freed neighbours are
// joined again, so memory freed at one size serves requests of any other. Only addresses are managed, the memory
// itself is never touched, which is what a GPU allocator needs (and what lets it be tested without one).
//
// Free blocks sit in lists by size class, 16 classes per power of two, found through bitmaps. Requests up to 1 MB
// come out of 2 MB segments of their own, so that small, long-lived tensors do not pin large segments.
// Not thread-safe.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

class BlockPool {
public:
    // allocate(bytes) asks for a segment and returns null when there is no memory; release gives one back
    BlockPool(std::function<void*(size_t)> allocate, std::function<void(void*)> release)
        : allocate_(std::move(allocate)), release_(std::move(release)) {
        live_.reserve(4096);
    }
    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;
    // Segments still out belong to the driver's context, which takes them back when the process ends
    ~BlockPool() {
        for (auto& b : live_) delete b.second;
        for (Arena* a : {&small_, &large_})
            for (auto& level : a->lists)
                for (Block* b : level)
                    while (b) {
                        Block* next = b->next;
                        delete b;
                        b = next;
                    }
        for (Block* b : spare_) delete b;
    }

    // Aligned to 512 bytes; null when there is no memory
    void* allocate(size_t bytes) {
        size_t size = round_up(bytes > 0 ? bytes : 1, kQuantum);
        bool small = size <= kSmallLimit;
        Arena& arena = small ? small_ : large_;
        Block* b = find(arena, size);
        if (b) {
            remove(arena, b);
        } else if (!(b = new_segment(size, small))) {
            return nullptr;
        }
        // What is left over stays available, unless it is too little to be worth it
        size_t rest = b->size - size;
        if (small ? rest >= kQuantum : rest > kSmallLimit) {
            Block* r = new_block();
            r->ptr = b->ptr + size;
            r->size = rest;
            r->small = small;
            r->left = b;
            r->right = b->right;
            if (r->right) r->right->left = r;
            b->right = r;
            b->size = size;
            insert(arena, r);
        }
        live_[b->ptr] = b;
        in_use_ += b->size;
        return b->ptr;
    }

    // False if ptr is not a live block of this pool
    bool free(void* ptr) {
        auto it = live_.find(ptr);
        if (it == live_.end()) return false;
        Block* b = it->second;
        live_.erase(it);
        in_use_ -= b->size;
        Arena& arena = b->small ? small_ : large_;
        if (b->left && b->left->free) {
            Block* l = b->left;
            remove(arena, l);
            join(l, b);
            b = l;
        }
        if (b->right && b->right->free) {
            Block* r = b->right;
            remove(arena, r);
            join(b, r);
        }
        insert(arena, b);
        return true;
    }

    // Gives back every segment nothing uses and returns how many bytes that was
    size_t release_unused() {
        size_t bytes = 0;
        for (Arena* arena : {&small_, &large_}) {
            std::vector<Block*> whole;
            for (auto& level : arena->lists)
                for (Block* b : level)
                    for (; b; b = b->next) if (!b->left && !b->right) whole.push_back(b);
            for (Block* b : whole) {
                remove(*arena, b);
                release_(b->ptr);
                bytes += b->size;
                recycle(b);
            }
        }
        reserved_ -= bytes;
        return bytes;
    }

    size_t in_use() const { return in_use_; }      // in live blocks
    size_t reserved() const { return reserved_; }  // in segments
    static size_t round_up(size_t bytes, size_t to) { return (bytes + to - 1) / to * to; }

    static constexpr size_t kQuantum = 512;
    static constexpr size_t kSmallLimit = (size_t)1 << 20;
    static constexpr size_t kSmallSegment = (size_t)2 << 20;
    static constexpr size_t kLargeSegment = (size_t)20 << 20;

private:
    struct Block {
        char* ptr = nullptr;
        size_t size = 0;
        Block *left = nullptr, *right = nullptr;  // neighbours in the segment
        Block *prev = nullptr, *next = nullptr;   // in a free list
        bool free = false, small = false;
    };
    struct Arena {
        Block* lists[64][16] = {};
        uint64_t levels = 0;     // bit i: some list of lists[i] has blocks
        uint16_t subs[64] = {};  // bit j of subs[i]: lists[i][j] has blocks
    };

    static int low_bit(uint64_t v) {
        int n = 0;
        while (!(v & 1)) {
            // Skips whole bytes first
            if (!(v & 0xff)) {
                v >>= 8;
                n += 8;
            } else {
                v >>= 1;
                ++n;
            }
        }
        return n;
    }
    static int high_bit(uint64_t v) {
        int n = 0;
        for (int step = 32; step > 0; step >>= 1) {
            if (v >> step) {
                v >>= step;
                n += step;
            }
        }
        return n;
    }
    static std::pair<int, int> size_class(size_t size) {
        int level = high_bit((uint64_t)size);  // at least 9
        return {level, (int)((size >> (level - 4)) & 15)};
    }

    void insert(Arena& a, Block* b) {
        auto [i, j] = size_class(b->size);
        b->free = true;
        b->prev = nullptr;
        b->next = a.lists[i][j];
        if (b->next) b->next->prev = b;
        a.lists[i][j] = b;
        a.subs[i] = (uint16_t)(a.subs[i] | (1u << j));
        a.levels |= (uint64_t)1 << i;
    }
    void remove(Arena& a, Block* b) {
        auto [i, j] = size_class(b->size);
        if (b->prev) b->prev->next = b->next;
        else a.lists[i][j] = b->next;
        if (b->next) b->next->prev = b->prev;
        if (!a.lists[i][j]) {
            a.subs[i] = (uint16_t)(a.subs[i] & ~(1u << j));
            if (!a.subs[i]) a.levels &= ~((uint64_t)1 << i);
        }
        b->free = false;
        b->prev = b->next = nullptr;
    }

    Block* find(Arena& a, size_t size) {
        auto [i, j] = size_class(size);
        // Blocks of the request's own class may or may not be large enough: the best of the first few
        Block* best = nullptr;
        int looked = 0;
        for (Block* b = a.lists[i][j]; b && looked < 16; b = b->next, ++looked) {
            if (b->size < size || (best && b->size >= best->size)) continue;
            best = b;
            if (b->size == size) break;
        }
        if (best) return best;
        // Every block of a later class is: one from the lowest that has any
        unsigned later = (unsigned)a.subs[i] >> (j + 1) << (j + 1);
        if (later) return a.lists[i][low_bit(later)];
        uint64_t higher = i + 1 < 64 ? a.levels >> (i + 1) << (i + 1) : 0;
        if (!higher) return nullptr;
        int k = low_bit(higher);
        return a.lists[k][low_bit(a.subs[k])];
    }

    Block* new_segment(size_t size, bool small) {
        size_t bytes = small ? kSmallSegment : size <= kLargeSegment / 2 ? kLargeSegment : round_up(size, kSmallSegment);
        void* p = allocate_(bytes);
        if (!p && bytes > round_up(size, kSmallSegment)) {
            bytes = round_up(size, kSmallSegment);
            p = allocate_(bytes);
        }
        if (!p) return nullptr;
        Block* b = new_block();
        b->ptr = static_cast<char*>(p);
        b->size = bytes;
        b->small = small;
        reserved_ += bytes;
        return b;
    }
    // Absorbs r, b's right neighbour
    void join(Block* b, Block* r) {
        b->size += r->size;
        b->right = r->right;
        if (b->right) b->right->left = b;
        recycle(r);
    }
    Block* new_block() {
        if (spare_.empty()) return new Block;
        Block* b = spare_.back();
        spare_.pop_back();
        return b;
    }
    void recycle(Block* b) {
        *b = Block();
        spare_.push_back(b);
    }

    std::function<void*(size_t)> allocate_;
    std::function<void(void*)> release_;
    Arena small_, large_;
    std::unordered_map<void*, Block*> live_;
    std::vector<Block*> spare_;
    size_t in_use_ = 0, reserved_ = 0;
};
