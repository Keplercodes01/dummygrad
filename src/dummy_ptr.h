#pragma once
#include <atomic>

struct DummyIntrusiveBase {
    mutable std::atomic<uint32_t> ref_count{0};
    virtual ~DummyIntrusiveBase() = default; 

    //called when a pointer is copied
    void retain() const noexcept {
        ref_count.fetch_add(1, std::memory_order_relaxed);
    }

    //called when a pointer is destroyed
    bool release() const noexcept {
        uint32_t old_value = ref_count.fetch_sub(1, std::memory_order_release);
        if(old_value == 1) {
            //hardware barrier to stop dummy_ptr wrapper from deleting this object before all the other threads have finished computation       
            std::atomic_thread_fence(std::memory_order_acquire);
            return true;
        }
        else
            return false;
    }
};

template <typename T>  
class dummy_ptr {
private:
    T* _ptr;
    //if the pointer isn't null, increment the counter
    void retain_if_valid() const noexcept {
        if(_ptr) 
            _ptr->retain();
    }
    //if the pointer isn't null, drop the counter. if it hits 0, delete it
    void release_if_valid() noexcept {
        if(_ptr) {
            if(_ptr->release())
                delete _ptr; //free the memory
        }
        _ptr = nullptr;
    }

public:
    //default constructor
    dummy_ptr() noexcept : _ptr(nullptr) {}
    //construct from a raw pointer
    explicit dummy_ptr(T* p) noexcept : _ptr(p)  {
        retain_if_valid();
    }
    //destructor
    ~dummy_ptr() {
        release_if_valid();
    }
    //copy constructor
    dummy_ptr(const dummy_ptr& other) noexcept : _ptr(other._ptr) {
        retain_if_valid();
    }

    //overloading
    T* get() const noexcept { return _ptr; }
    T* opearator->() const noexcept { return _ptr; }
    T& opearator*() const noexcept { return *_ptr; }
    explicit opearator bool() const noexcept {
        return _ptr != nullptr; 
    }
};
