#include "pjrt_client.h"
#include <iostream>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <vector>

TPUBufferHandle::~TPUBufferHandle() {
    if (buffer) {
        PJRTTPUManager::get().free_buffer(buffer);
        buffer = nullptr;
    }
}

TPUBufferHandle::TPUBufferHandle(TPUBufferHandle&& other) noexcept
    : buffer(other.buffer), device_id(other.device_id),
      total_bytes(other.total_bytes), dims(std::move(other.dims)), dtype(other.dtype) {
    other.buffer = nullptr;
}

TPUBufferHandle& TPUBufferHandle::operator=(TPUBufferHandle&& other) noexcept {
    if (this != &other) {
        if (buffer) {
            PJRTTPUManager::get().free_buffer(buffer);
        }
        buffer = other.buffer;
        device_id = other.device_id;
        total_bytes = other.total_bytes;
        dims = std::move(other.dims);
        dtype = other.dtype;
        other.buffer = nullptr;
    }
    return *this;
}

static PJRT_Buffer_Type to_pjrt_buffer_type(DType dt) {
    switch (dt) {
        case DType::Float32:  return PJRT_Buffer_Type_F32;
        case DType::Float16:  return PJRT_Buffer_Type_F16;
        case DType::BFloat16: return PJRT_Buffer_Type_BF16;
        case DType::Int8:     return PJRT_Buffer_Type_S8;
        default:              return PJRT_Buffer_Type_F32;
    }
}

PJRTTPUManager::PJRTTPUManager() {
    const char* candidate_paths[] = {
        std::getenv("TPU_LIBRARY_PATH"),
        std::getenv("PJRT_TPU_LIBRARY_PATH"),
        "/usr/local/lib/python3.10/dist-packages/libtpu/libtpu.so",
        "/usr/local/lib/python3.11/dist-packages/libtpu/libtpu.so",
        "/usr/local/lib/python3.12/dist-packages/libtpu/libtpu.so",
        "/opt/conda/lib/python3.10/site-packages/libtpu/libtpu.so",
        "/usr/lib/libtpu.so",
        "/lib/libtpu.so",
        "libpjrt_tpu.so",
        "libtpu.so"
    };

    for (const char* path : candidate_paths) {
        if (!path) continue;
        lib_handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (lib_handle) {
            break;
        }
    }

    if (!lib_handle) {
        init_error = "OpenXLA PJRT: Could not find or load libtpu.so / libpjrt_tpu.so.";
        return;
    }

    load_symbols();
}

PJRTTPUManager::~PJRTTPUManager() {
    if (client && fn_client_destroy) {
        fn_client_destroy(client);
        client = nullptr;
    }
    if (lib_handle) {
        dlclose(lib_handle);
        lib_handle = nullptr;
    }
}

template <typename T>
static bool load_sym(void* handle, T& func_ptr, const char* name) {
    func_ptr = reinterpret_cast<T>(dlsym(handle, name));
    return func_ptr != nullptr;
}

void PJRTTPUManager::load_symbols() {
    bool ok = true;
    ok &= load_sym(lib_handle, fn_client_create, "PJRT_Client_Create");
    ok &= load_sym(lib_handle, fn_client_destroy, "PJRT_Client_Destroy");
    ok &= load_sym(lib_handle, fn_addressable_devices, "PJRT_Client_AddressableDevices");
    ok &= load_sym(lib_handle, fn_client_compile, "PJRT_Client_Compile");
    ok &= load_sym(lib_handle, fn_executable_execute, "PJRT_LoadedExecutable_Execute");
    ok &= load_sym(lib_handle, fn_executable_destroy, "PJRT_LoadedExecutable_Destroy");
    ok &= load_sym(lib_handle, fn_buffer_from_host, "PJRT_Client_BufferFromHostBuffer");
    ok &= load_sym(lib_handle, fn_buffer_to_host, "PJRT_Buffer_ToHostBuffer");
    ok &= load_sym(lib_handle, fn_buffer_destroy, "PJRT_Buffer_Destroy");
    ok &= load_sym(lib_handle, fn_event_await, "PJRT_Event_Await");
    ok &= load_sym(lib_handle, fn_event_destroy, "PJRT_Event_Destroy");
    ok &= load_sym(lib_handle, fn_error_message, "PJRT_Error_GetMessage");
    ok &= load_sym(lib_handle, fn_error_destroy, "PJRT_Error_Destroy");

    if (!ok) {
        init_error = "OpenXLA PJRT: Failed to resolve one or more C API symbols in TPU shared library.";
        dlclose(lib_handle);
        lib_handle = nullptr;
        return;
    }

    PJRT_Client_Create_Args args;
    std::memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    PJRT_Error* err = fn_client_create(&args);
    if (err) {
        check_error(err, "PJRT_Client_Create");
        return;
    }
    client = args.client;

    // Enumerate addressable TPU devices (v5e-1 has 1 chip, v5e-8 has 8 chips)
    PJRT_Device** dev_ptrs = nullptr;
    size_t num_devs = 0;
    err = fn_addressable_devices(client, &dev_ptrs, &num_devs);
    if (err) {
        check_error(err, "PJRT_Client_AddressableDevices");
        return;
    }

    for (size_t i = 0; i < num_devs; ++i) {
        devices.push_back(dev_ptrs[i]);
    }

    initialized = true;
}

void PJRTTPUManager::check_error(PJRT_Error* err, const char* context) {
    if (!err) return;
    const char* msg = "Unknown PJRT error";
    size_t len = 0;
    if (fn_error_message) {
        fn_error_message(err, &msg, &len);
    }
    std::string err_str = std::string(context) + " failed: " + (msg ? msg : "null");
    if (fn_error_destroy) {
        fn_error_destroy(err);
    }
    throw std::runtime_error(err_str);
}

std::shared_ptr<TPUBufferHandle> PJRTTPUManager::create_buffer_from_host(
    const void* host_ptr,
    const std::vector<int64_t>& shape,
    DType dtype,
    size_t device_id
) {
    if (!initialized) {
        throw std::runtime_error("PJRT TPU runtime not initialized: " + init_error);
    }
    if (device_id >= devices.size()) {
        throw std::runtime_error("PJRT TPU: Invalid device_id " + std::to_string(device_id));
    }

    size_t elem_size = dtype_size(dtype);
    size_t num_elements = 1;
    for (auto d : shape) num_elements *= d;
    size_t total_bytes = num_elements * elem_size;

    std::vector<int64_t> byte_strides(shape.size());
    int64_t stride = elem_size;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        byte_strides[i] = stride;
        stride *= shape[i];
    }

    PJRT_Client_BufferFromHostBuffer_Args args;
    std::memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.client = client;
    args.data = host_ptr;
    args.type = to_pjrt_buffer_type(dtype);
    args.dims = shape.data();
    args.num_dims = shape.size();
    args.byte_strides = byte_strides.data();
    args.num_byte_strides = byte_strides.size();
    args.device = devices[device_id];

    PJRT_Error* err = fn_buffer_from_host(&args);
    if (err) {
        check_error(err, "PJRT_Client_BufferFromHostBuffer");
    }

    if (args.done_event) {
        fn_event_await(args.done_event);
        fn_event_destroy(args.done_event);
    }

    return std::make_shared<TPUBufferHandle>(args.buffer, device_id, total_bytes, shape, dtype);
}

std::shared_ptr<TPUBufferHandle> PJRTTPUManager::allocate_device_buffer(
    const std::vector<int64_t>& shape,
    DType dtype,
    size_t device_id
) {
    size_t elem_size = dtype_size(dtype);
    size_t num_elements = 1;
    for (auto d : shape) num_elements *= d;
    size_t total_bytes = num_elements * elem_size;

    std::vector<uint8_t> zero_host(total_bytes, 0);
    return create_buffer_from_host(zero_host.data(), shape, dtype, device_id);
}

void PJRTTPUManager::copy_to_host(const TPUBufferHandle& handle, void* host_dst, size_t bytes) {
    if (!initialized || !handle.buffer) {
        throw std::runtime_error("PJRT TPU: Invalid buffer handle or uninitialized runtime");
    }

    PJRT_Buffer_ToHostBuffer_Args args;
    std::memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.buffer = handle.buffer;
    args.dst = host_dst;
    args.dst_size = bytes;

    PJRT_Error* err = fn_buffer_to_host(&args);
    if (err) {
        check_error(err, "PJRT_Buffer_ToHostBuffer");
    }

    if (args.event) {
        fn_event_await(args.event);
        fn_event_destroy(args.event);
    }
}

void PJRTTPUManager::free_buffer(PJRT_Buffer* buffer) {
    if (buffer && fn_buffer_destroy) {
        fn_buffer_destroy(buffer);
    }
}

PJRT_LoadedExecutable* PJRTTPUManager::compile_hlo(const std::string& hlo_code) {
    if (!initialized) {
        throw std::runtime_error("PJRT TPU not initialized: " + init_error);
    }

    PJRT_Program program;
    std::memset(&program, 0, sizeof(program));
    program.struct_size = sizeof(program);
    program.format = "hlo";
    program.format_size = 3;
    program.code = hlo_code.c_str();
    program.code_size = hlo_code.size();

    PJRT_Client_Compile_Args args;
    std::memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.client = client;
    args.program = &program;
    args.compile_options = nullptr;

    PJRT_Error* err = fn_client_compile(&args);
    if (err) {
        check_error(err, "PJRT_Client_Compile");
    }

    return args.executable;
}

std::vector<std::shared_ptr<TPUBufferHandle>> PJRTTPUManager::execute(
    PJRT_LoadedExecutable* executable,
    const std::vector<std::shared_ptr<TPUBufferHandle>>& inputs,
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<DType>& output_dtypes,
    size_t device_id
) {
    if (!initialized || !executable) {
        throw std::runtime_error("PJRT TPU: Invalid executable or uninitialized client");
    }

    std::vector<PJRT_Buffer*> raw_inputs;
    raw_inputs.reserve(inputs.size());
    for (const auto& inp : inputs) {
        raw_inputs.push_back(inp->buffer);
    }

    PJRT_Buffer* const* device_arg_list = raw_inputs.data();
    PJRT_Buffer* const* const arg_lists[1] = { device_arg_list };

    PJRT_ExecuteOptions options;
    std::memset(&options, 0, sizeof(options));
    options.struct_size = sizeof(options);

    PJRT_LoadedExecutable_Execute_Args args;
    std::memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.executable = executable;
    args.options = &options;
    args.argument_lists = arg_lists;
    args.num_devices = 1;
    args.num_args = inputs.size();

    PJRT_Error* err = fn_executable_execute(&args);
    if (err) {
        check_error(err, "PJRT_LoadedExecutable_Execute");
    }

    // Await completion events
    if (args.complete_events && args.complete_events[0] && args.complete_events[0][0]) {
        fn_event_await(args.complete_events[0][0]);
        fn_event_destroy(args.complete_events[0][0]);
    }

    std::vector<std::shared_ptr<TPUBufferHandle>> outputs;
    outputs.reserve(output_shapes.size());
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        size_t elem_size = dtype_size(output_dtypes[i]);
        size_t n = 1;
        for (auto d : output_shapes[i]) n *= d;
        outputs.push_back(std::make_shared<TPUBufferHandle>(
            args.output_lists[0][i],
            device_id,
            n * elem_size,
            output_shapes[i],
            output_dtypes[i]
        ));
    }

    return outputs;
}

void PJRTTPUManager::destroy_executable(PJRT_LoadedExecutable* executable) {
    if (executable && fn_executable_destroy) {
        fn_executable_destroy(executable);
    }
}
