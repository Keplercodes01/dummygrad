#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = (call); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - code " + std::to_string(status)); \
        } \
    } while (0)

inline cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        CUBLAS_CHECK(cublasCreate(&handle));
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return handle;
}

// -------------------------------------------------------------
// CUDA Graph Execution (Hardware-level step capture & replay)
// Eliminates 100% of CPU dispatch latency across training steps
// -------------------------------------------------------------
struct CUDAGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    bool is_captured = false;

    void begin_capture(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        is_captured = false;
    }

    void end_capture(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
        is_captured = true;
    }

    void replay(cudaStream_t stream = nullptr) {
        if (!is_captured || !instance) {
            throw std::runtime_error("CUDAGraph: graph not captured yet");
        }
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
    }

    ~CUDAGraph() {
        if (instance) cudaGraphExecDestroy(instance);
        if (graph) cudaGraphDestroy(graph);
    }
};

#endif
