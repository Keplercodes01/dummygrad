#pragma once
#include <cstdint>
#include <cstddef>

// Standard OpenXLA PJRT C API types and structs
// Enables dynamic loading of libtpu.so / libpjrt_tpu.so without proprietary SDK build-time dependencies

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PJRT_Error PJRT_Error;
typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Buffer PJRT_Buffer;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Event PJRT_Event;

typedef enum PJRT_Buffer_Type {
    PJRT_Buffer_Type_INVALID = 0,
    PJRT_Buffer_Type_PRED = 1,
    PJRT_Buffer_Type_S8 = 2,
    PJRT_Buffer_Type_S16 = 3,
    PJRT_Buffer_Type_S32 = 4,
    PJRT_Buffer_Type_S64 = 5,
    PJRT_Buffer_Type_U8 = 6,
    PJRT_Buffer_Type_U16 = 7,
    PJRT_Buffer_Type_U32 = 8,
    PJRT_Buffer_Type_U64 = 9,
    PJRT_Buffer_Type_F16 = 10,
    PJRT_Buffer_Type_F32 = 11,
    PJRT_Buffer_Type_F64 = 12,
    PJRT_Buffer_Type_BF16 = 13,
    PJRT_Buffer_Type_C64 = 14,
    PJRT_Buffer_Type_C128 = 15,
} PJRT_Buffer_Type;

typedef enum PJRT_ErrorCode {
    PJRT_Error_Code_OK = 0,
    PJRT_Error_Code_CANCELLED = 1,
    PJRT_Error_Code_UNKNOWN = 2,
    PJRT_Error_Code_INVALID_ARGUMENT = 3,
    PJRT_Error_Code_DEADLINE_EXCEEDED = 4,
    PJRT_Error_Code_NOT_FOUND = 5,
    PJRT_Error_Code_ALREADY_EXISTS = 6,
    PJRT_Error_Code_PERMISSION_DENIED = 7,
    PJRT_Error_Code_RESOURCE_EXHAUSTED = 8,
    PJRT_Error_Code_FAILED_PRECONDITION = 9,
    PJRT_Error_Code_ABORTED = 10,
    PJRT_Error_Code_OUT_OF_RANGE = 11,
    PJRT_Error_Code_UNIMPLEMENTED = 12,
    PJRT_Error_Code_INTERNAL = 13,
    PJRT_Error_Code_UNAVAILABLE = 14,
    PJRT_Error_Code_DATA_LOSS = 15,
    PJRT_Error_Code_UNAUTHENTICATED = 16,
} PJRT_ErrorCode;

struct PJRT_Client_Create_Args {
    size_t struct_size;
    void* extension_start;
    PJRT_Client* client;
};

struct PJRT_Compile_Options {
    size_t struct_size;
    void* extension_start;
    const char* compile_options;
    size_t compile_options_size;
};

struct PJRT_Program {
    size_t struct_size;
    void* extension_start;
    const char* format;
    size_t format_size;
    const char* code;
    size_t code_size;
};

struct PJRT_Client_Compile_Args {
    size_t struct_size;
    void* extension_start;
    PJRT_Client* client;
    const struct PJRT_Program* program;
    const struct PJRT_Compile_Options* compile_options;
    PJRT_LoadedExecutable* executable;
};

struct PJRT_ExecuteOptions {
    size_t struct_size;
    void* extension_start;
    bool non_donatable_input_indices;
};

struct PJRT_LoadedExecutable_Execute_Args {
    size_t struct_size;
    void* extension_start;
    PJRT_LoadedExecutable* executable;
    const struct PJRT_ExecuteOptions* options;
    PJRT_Buffer* const* const* argument_lists;
    size_t num_devices;
    size_t num_args;
    PJRT_Buffer*** output_lists;
    PJRT_Event*** complete_events;
};

struct PJRT_Client_BufferFromHostBuffer_Args {
    size_t struct_size;
    void* extension_start;
    PJRT_Client* client;
    const void* data;
    PJRT_Buffer_Type type;
    const int64_t* dims;
    size_t num_dims;
    const int64_t* byte_strides;
    size_t num_byte_strides;
    PJRT_Device* device;
    PJRT_Buffer* buffer;
    PJRT_Event* done_event;
};

struct PJRT_Buffer_ToHostBuffer_Args {
    size_t struct_size;
    void* extension_start;
    PJRT_Buffer* buffer;
    void* dst;
    size_t dst_size;
    PJRT_Event* event;
};

typedef PJRT_Error* (*PJRT_Client_Create_Fn)(struct PJRT_Client_Create_Args* args);
typedef PJRT_Error* (*PJRT_Client_Destroy_Fn)(PJRT_Client* client);
typedef PJRT_Error* (*PJRT_Client_Devices_Fn)(PJRT_Client* client, PJRT_Device*** devices, size_t* num_devices);
typedef PJRT_Error* (*PJRT_Client_AddressableDevices_Fn)(PJRT_Client* client, PJRT_Device*** devices, size_t* num_devices);
typedef PJRT_Error* (*PJRT_Client_Compile_Fn)(struct PJRT_Client_Compile_Args* args);
typedef PJRT_Error* (*PJRT_LoadedExecutable_Execute_Fn)(struct PJRT_LoadedExecutable_Execute_Args* args);
typedef PJRT_Error* (*PJRT_LoadedExecutable_Destroy_Fn)(PJRT_LoadedExecutable* executable);
typedef PJRT_Error* (*PJRT_Client_BufferFromHostBuffer_Fn)(struct PJRT_Client_BufferFromHostBuffer_Args* args);
typedef PJRT_Error* (*PJRT_Buffer_ToHostBuffer_Fn)(struct PJRT_Buffer_ToHostBuffer_Args* args);
typedef PJRT_Error* (*PJRT_Buffer_Destroy_Fn)(PJRT_Buffer* buffer);
typedef PJRT_Error* (*PJRT_Event_Await_Fn)(PJRT_Event* event);
typedef PJRT_Error* (*PJRT_Event_Destroy_Fn)(PJRT_Event* event);
typedef void (*PJRT_Error_GetMessage_Fn)(PJRT_Error* error, const char** message, size_t* message_size);
typedef PJRT_ErrorCode (*PJRT_Error_GetCode_Fn)(PJRT_Error* error);
typedef void (*PJRT_Error_Destroy_Fn)(PJRT_Error* error);

#ifdef __cplusplus
}
#endif
