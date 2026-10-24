#pragma once
// Device-side block reductions shared by the CUDA kernel files. Device code only: include this
// from .cu files, never from headers that host compilers see.

#ifdef __CUDACC__

namespace cuda {

// Block Reduction Primitives (Warp Shuffle + Shared Memory)
// Reduces values across all threads in a block (up to 1024 threads)
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float s_warp_sum[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (lane_id == 0) {
        s_warp_sum[warp_id] = val;
    }
    __syncthreads();

    float warp_val = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp_sum[lane_id] : 0.0f;
    if (warp_id == 0) {
        unsigned int mask0 = __activemask();
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_val += __shfl_down_sync(mask0, warp_val, offset);
        }
    }
    __syncthreads();
    return warp_val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float s_warp_max[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }

    if (lane_id == 0) {
        s_warp_max[warp_id] = val;
    }
    __syncthreads();

    float warp_val = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp_max[lane_id] : -1e20f;
    if (warp_id == 0) {
        unsigned int mask0 = __activemask();
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_val = fmaxf(warp_val, __shfl_down_sync(mask0, warp_val, offset));
        }
    }
    __syncthreads();
    return warp_val;
}

}  // namespace cuda

#endif  // __CUDACC__
