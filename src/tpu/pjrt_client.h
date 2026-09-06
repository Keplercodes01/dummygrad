#pragma once
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <stdexcept>
#include "../types.h"
#include "pjrt_c_api.h"

// RAII Handle for TPU Device Buffer managed by OpenXLA PJRT
struct TPUBufferHandle {
    PJRT_Buffer* buffer = nullptr;
    size_t device_id = 0;
    size_t total_bytes = 0;
    std::vector<int64_t> dims;
    DType dtype = DType::Float32;

    TPUBufferHandle() = default;
    TPUBufferHandle(PJRT_Buffer* buf, size_t dev_id, size_t bytes, const std::vector<int64_t>& s, DType dt)
        : buffer(buf), device_id(dev_id), total_bytes(bytes), dims(s), dtype(dt) {}
    
    ~TPUBufferHandle();
    TPUBufferHandle(const TPUBufferHandle&) = delete;
    TPUBufferHandle& operator=(const TPUBufferHandle&) = delete;
    TPUBufferHandle(TPUBufferHandle&& other) noexcept;
    TPUBufferHandle& operator=(TPUBufferHandle&& other) noexcept;
};

// Manager for OpenXLA PJRT TPU Runtime
// Dynamically discovers and interacts with TPU hardware (Google Colab v5e-1 & Kaggle v5e-8)
class PJRTTPUManager {
private:
    void* lib_handle = nullptr;
    PJRT_Client* client = nullptr;
    std::vector<PJRT_Device*> devices;
    bool initialized = false;
    std::string init_error;
    std::mutex mtx;

    // PJRT Function Pointers
    PJRT_Client_Create_Fn fn_client_create = nullptr;
    PJRT_Client_Destroy_Fn fn_client_destroy = nullptr;
    PJRT_Client_AddressableDevices_Fn fn_addressable_devices = nullptr;
    PJRT_Client_Compile_Fn fn_client_compile = nullptr;
    PJRT_LoadedExecutable_Execute_Fn fn_executable_execute = nullptr;
    PJRT_LoadedExecutable_Destroy_Fn fn_executable_destroy = nullptr;
    PJRT_Client_BufferFromHostBuffer_Fn fn_buffer_from_host = nullptr;
    PJRT_Buffer_ToHostBuffer_Fn fn_buffer_to_host = nullptr;
    PJRT_Buffer_Destroy_Fn fn_buffer_destroy = nullptr;
    PJRT_Event_Await_Fn fn_event_await = nullptr;
    PJRT_Event_Destroy_Fn fn_event_destroy = nullptr;
    PJRT_Error_GetMessage_Fn fn_error_message = nullptr;
    PJRT_Error_Destroy_Fn fn_error_destroy = nullptr;

    PJRTTPUManager();
    ~PJRTTPUManager();
    void load_symbols();
    void check_error(PJRT_Error* err, const char* context);

public:
    static PJRTTPUManager& get() {
        static PJRTTPUManager instance;
        return instance;
    }

    bool is_available() const { return initialized; }
    const std::string& error_message() const { return init_error; }
    size_t device_count() const { return devices.size(); }

    // Allocate TPU buffer from host memory
    std::shared_ptr<TPUBufferHandle> create_buffer_from_host(
        const void* host_ptr,
        const std::vector<int64_t>& shape,
        DType dtype,
        size_t device_id = 0
    );

    // Allocate uninitialized TPU buffer
    std::shared_ptr<TPUBufferHandle> allocate_device_buffer(
        const std::vector<int64_t>& shape,
        DType dtype,
        size_t device_id = 0
    );

    // Transfer data from TPU buffer back to host memory
    void copy_to_host(const TPUBufferHandle& handle, void* host_dst, size_t bytes);

    // Free TPU buffer
    void free_buffer(PJRT_Buffer* buffer);

    // Compile HLO computation graph into TPU executable
    PJRT_LoadedExecutable* compile_hlo(const std::string& hlo_code);

    // Execute compiled HLO on specified device
    std::vector<std::shared_ptr<TPUBufferHandle>> execute(
        PJRT_LoadedExecutable* executable,
        const std::vector<std::shared_ptr<TPUBufferHandle>>& inputs,
        const std::vector<std::vector<int64_t>>& output_shapes,
        const std::vector<DType>& output_dtypes,
        size_t device_id = 0
    );

    void destroy_executable(PJRT_LoadedExecutable* executable);
};
