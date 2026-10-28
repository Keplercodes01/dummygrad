#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "types.h"
#include "autocast.h"
#include <stdexcept>
#include <string>
#include <unordered_map>

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
    int dev = 0;
    cudaGetDevice(&dev);
    static thread_local std::unordered_map<int, cublasHandle_t> handles;
    auto it = handles.find(dev);
    cublasHandle_t handle = nullptr;
    if (it == handles.end()) {
        CUBLAS_CHECK(cublasCreate(&handle));
        handles[dev] = handle;
    } else {
        handle = it->second;
    }
    // TF32 tensor cores for Float32 GEMMs only when allowed (set_allow_tf32)
    CUBLAS_CHECK(cublasSetMathMode(handle, allow_tf32() ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
    return handle;
}

// cuBLAS strided-batched GEMM, column-major: C = alpha op_a(A) op_b(B) + beta C. C is fp32; A and
// B are fp32, or both of the compute dtype: Float16/BFloat16 run on tensor cores and accumulate in
// fp32. Float32 runs SGEMM, on TF32 tensor cores only with set_allow_tf32(true) (see the math mode
// above). batch == 1 ignores the strides.
inline void gemm_strided_batched(cublasOperation_t op_a, cublasOperation_t op_b, int m, int n, int k, float alpha,
                                 const void* A, int lda, long long stride_a, const void* B, int ldb, long long stride_b,
                                 float beta, float* C, int ldc, long long stride_c, int batch, DType compute) {
    cublasHandle_t handle = get_cublas_handle();
    if (compute == DType::Float32) {
        const float* a = static_cast<const float*>(A);
        const float* b = static_cast<const float*>(B);
        if (batch == 1) {
            CUBLAS_CHECK(cublasSgemm(handle, op_a, op_b, m, n, k, &alpha, a, lda, b, ldb, &beta, C, ldc));
        } else {
            CUBLAS_CHECK(cublasSgemmStridedBatched(handle, op_a, op_b, m, n, k, &alpha, a, lda, stride_a,
                                                   b, ldb, stride_b, &beta, C, ldc, stride_c, batch));
        }
        return;
    }
    if (compute != DType::Float16 && compute != DType::BFloat16) {
        throw std::runtime_error("gemm_strided_batched: compute dtype must be Float32, Float16 or BFloat16");
    }
    cudaDataType_t type = compute == DType::Float16 ? CUDA_R_16F : CUDA_R_16BF;
    if (batch == 1) {
        CUBLAS_CHECK(cublasGemmEx(handle, op_a, op_b, m, n, k, &alpha, A, type, lda, B, type, ldb,
                                  &beta, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else {
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, op_a, op_b, m, n, k, &alpha, A, type, lda, stride_a,
                                                B, type, ldb, stride_b, &beta, C, CUDA_R_32F, ldc, stride_c,
                                                batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
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
