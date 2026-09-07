#include "metal_backend.h"

#if defined(__APPLE__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <unordered_map>
#include <mutex>
#include <cmath>
#include <iostream>

class MetalBackendImpl : public MetalBackend {
private:
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> library = nil;

    id<MTLComputePipelineState> pipeline_bias_gelu = nil;
    id<MTLComputePipelineState> pipeline_bias_relu = nil;
    id<MTLComputePipelineState> pipeline_layernorm = nil;
    id<MTLComputePipelineState> pipeline_adamw = nil;
    id<MTLComputePipelineState> pipeline_attention = nil;

    id<MTLComputePipelineState> pipeline_sigmoid = nil;
    id<MTLComputePipelineState> pipeline_sigmoid_bwd = nil;
    id<MTLComputePipelineState> pipeline_silu = nil;
    id<MTLComputePipelineState> pipeline_silu_bwd = nil;
    id<MTLComputePipelineState> pipeline_leaky_relu = nil;
    id<MTLComputePipelineState> pipeline_leaky_relu_bwd = nil;
    id<MTLComputePipelineState> pipeline_mse_bwd = nil;
    id<MTLComputePipelineState> pipeline_l1_loss_bwd = nil;
    id<MTLComputePipelineState> pipeline_causal_mask = nil;
    id<MTLComputePipelineState> pipeline_causal_mask_bwd = nil;
    id<MTLComputePipelineState> pipeline_sliding_window_mask = nil;
    id<MTLComputePipelineState> pipeline_sliding_window_mask_bwd = nil;

    std::unordered_map<void*, id<MTLBuffer>> buffer_map;
    std::mutex mtx;
    bool ready = false;

public:
    MetalBackendImpl() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) return;

            commandQueue = [device newCommandQueue];
            if (!commandQueue) return;

            // Load MSL Shader Source
            NSString* shaderSource = @R"(
                #include <metal_stdlib>
                using namespace metal;

                kernel void fused_add_bias_gelu_kernel(
                    device const float* in       [[buffer(0)]],
                    device const float* bias     [[buffer(1)]],
                    device float* out            [[buffer(2)]],
                    constant uint& total_elems   [[buffer(3)]],
                    constant uint& cols          [[buffer(4)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    uint c = id % cols;
                    float val = in[id] + bias[c];
                    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
                    constexpr float COEFF = 0.044715f;
                    float cube = val * val * val;
                    float inner = SQRT_2_OVER_PI * (val + COEFF * cube);
                    out[id] = 0.5f * val * (1.0f + metal::tanh(inner));
                }

                kernel void fused_add_bias_relu_kernel(
                    device const float* in       [[buffer(0)]],
                    device const float* bias     [[buffer(1)]],
                    device float* out            [[buffer(2)]],
                    constant uint& total_elems   [[buffer(3)]],
                    constant uint& cols          [[buffer(4)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    uint c = id % cols;
                    float val = in[id] + bias[c];
                    out[id] = metal::max(val, 0.0f);
                }

                kernel void fused_layernorm_kernel(
                    device const float* in       [[buffer(0)]],
                    device const float* gamma    [[buffer(1)]],
                    device const float* beta     [[buffer(2)]],
                    device float* out            [[buffer(3)]],
                    constant uint& cols          [[buffer(4)]],
                    constant float& eps          [[buffer(5)]],
                    uint row                     [[threadgroup_position_in_grid]],
                    uint tid                     [[thread_position_in_threadgroup]],
                    uint simd_lane               [[thread_index_in_simdgroup]],
                    uint simd_id                 [[simdgroup_index_in_threadgroup]],
                    uint t_per_tg                [[threads_per_threadgroup]]
                ) {
                    device const float* row_in = in + row * cols;
                    device float* row_out = out + row * cols;
                    float sum = 0.0f;
                    float sq_sum = 0.0f;
                    for (uint i = tid; i < cols; i += t_per_tg) {
                        float x = row_in[i];
                        sum += x;
                        sq_sum += x * x;
                    }
                    sum = simd_sum(sum);
                    sq_sum = simd_sum(sq_sum);

                    threadgroup float s_mean[32];
                    threadgroup float s_var[32];
                    if (simd_lane == 0) {
                        s_mean[simd_id] = sum;
                        s_var[simd_id] = sq_sum;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    uint num_simds = (t_per_tg + 31) / 32;
                    float total_sum = 0.0f;
                    float total_sq = 0.0f;
                    if (tid < num_simds) {
                        total_sum = s_mean[tid];
                        total_sq = s_var[tid];
                    }
                    total_sum = simd_sum(total_sum);
                    total_sq = simd_sum(total_sq);

                    threadgroup float row_mean;
                    threadgroup float row_rstd;
                    if (tid == 0) {
                        float mean = total_sum / float(cols);
                        float variance = (total_sq / float(cols)) - (mean * mean);
                        if (variance < 0.0f) variance = 0.0f;
                        row_mean = mean;
                        row_rstd = metal::rsqrt(variance + eps);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    float m = row_mean;
                    float r = row_rstd;
                    for (uint i = tid; i < cols; i += t_per_tg) {
                        float g = gamma ? gamma[i] : 1.0f;
                        float b = beta ? beta[i] : 0.0f;
                        row_out[i] = ((row_in[i] - m) * r) * g + b;
                    }
                }

                kernel void fused_adamw_kernel(
                    device float* p              [[buffer(0)]],
                    device const float* g        [[buffer(1)]],
                    device float* m              [[buffer(2)]],
                    device float* v              [[buffer(3)]],
                    constant float& lr           [[buffer(4)]],
                    constant float& beta1        [[buffer(5)]],
                    constant float& beta2        [[buffer(6)]],
                    constant float& eps          [[buffer(7)]],
                    constant float& weight_decay [[buffer(8)]],
                    constant float& bc1          [[buffer(9)]],
                    constant float& bc2          [[buffer(10)]],
                    constant uint& total_params  [[buffer(11)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_params) return;
                    float param = p[id];
                    float grad = g[id];
                    if (weight_decay != 0.0f) {
                        param -= lr * weight_decay * param;
                    }
                    float mom = beta1 * m[id] + (1.0f - beta1) * grad;
                    float var = beta2 * v[id] + (1.0f - beta2) * grad * grad;
                    m[id] = mom;
                    v[id] = var;
                    float mom_hat = mom / bc1;
                    float var_hat = var / bc2;
                    param -= (lr * mom_hat) / (metal::sqrt(var_hat) + eps);
                    p[id] = param;
                }

                kernel void tiled_flash_attention_kernel(
                    device const float* Q        [[buffer(0)]],
                    device const float* K        [[buffer(1)]],
                    device const float* V        [[buffer(2)]],
                    device float* Out            [[buffer(3)]],
                    constant uint& seq_len       [[buffer(4)]],
                    constant uint& head_dim      [[buffer(5)]],
                    constant float& scale        [[buffer(6)]],
                    uint2 tg_pos                 [[threadgroup_position_in_grid]],
                    uint tid                     [[thread_position_in_threadgroup]]
                ) {
                    uint head_id = tg_pos.y;
                    uint q_idx = tg_pos.x;
                    if (q_idx >= seq_len) return;
                    uint head_offset = head_id * seq_len * head_dim;
                    device const float* q_vec = Q + head_offset + q_idx * head_dim;
                    device float* out_vec = Out + head_offset + q_idx * head_dim;
                    float max_score = -1e20f;
                    float sum_exp = 0.0f;
                    for (uint k_idx = 0; k_idx <= q_idx; ++k_idx) {
                        device const float* k_vec = K + head_offset + k_idx * head_dim;
                        float dot = 0.0f;
                        for (uint d = 0; d < head_dim; ++d) dot += q_vec[d] * k_vec[d];
                        dot *= scale;
                        if (dot > max_score) {
                            sum_exp = sum_exp * metal::exp(max_score - dot) + 1.0f;
                            max_score = dot;
                        } else {
                            sum_exp += metal::exp(dot - max_score);
                        }
                    }
                    if (tid < head_dim) {
                        float acc = 0.0f;
                        for (uint k_idx = 0; k_idx <= q_idx; ++k_idx) {
                            device const float* k_vec = K + head_offset + k_idx * head_dim;
                            device const float* v_vec = V + head_offset + k_idx * head_dim;
                            float dot = 0.0f;
                            for (uint d = 0; d < head_dim; ++d) dot += q_vec[d] * k_vec[d];
                            dot *= scale;
                            float weight = metal::exp(dot - max_score) / sum_exp;
                            acc += weight * v_vec[tid];
                        }
                        out_vec[tid] = acc;
                    }
                }

                kernel void sigmoid_kernel(
                    device const float* in       [[buffer(0)]],
                    device float* out            [[buffer(1)]],
                    constant uint& total_elems   [[buffer(2)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    out[id] = 1.0f / (1.0f + metal::exp(-in[id]));
                }

                kernel void sigmoid_backward_kernel(
                    device const float* out      [[buffer(0)]],
                    device const float* grad_out [[buffer(1)]],
                    device float* grad_in        [[buffer(2)]],
                    constant uint& total_elems   [[buffer(3)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    float s = out[id];
                    grad_in[id] = s * (1.0f - s) * grad_out[id];
                }

                kernel void silu_kernel(
                    device const float* in       [[buffer(0)]],
                    device float* out            [[buffer(1)]],
                    constant uint& total_elems   [[buffer(2)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    float x = in[id];
                    out[id] = x / (1.0f + metal::exp(-x));
                }

                kernel void silu_backward_kernel(
                    device const float* in       [[buffer(0)]],
                    device const float* grad_out [[buffer(1)]],
                    device float* grad_in        [[buffer(2)]],
                    constant uint& total_elems   [[buffer(3)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    float x = in[id];
                    float s = 1.0f / (1.0f + metal::exp(-x));
                    grad_in[id] = (s * (1.0f + x * (1.0f - s))) * grad_out[id];
                }

                kernel void leaky_relu_kernel(
                    device const float* in          [[buffer(0)]],
                    device float* out               [[buffer(1)]],
                    constant float& negative_slope  [[buffer(2)]],
                    constant uint& total_elems      [[buffer(3)]],
                    uint id                         [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    float x = in[id];
                    out[id] = x > 0.0f ? x : negative_slope * x;
                }

                kernel void leaky_relu_backward_kernel(
                    device const float* in          [[buffer(0)]],
                    device const float* grad_out    [[buffer(1)]],
                    device float* grad_in           [[buffer(2)]],
                    constant float& negative_slope  [[buffer(3)]],
                    constant uint& total_elems      [[buffer(4)]],
                    uint id                         [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    grad_in[id] = in[id] > 0.0f ? grad_out[id] : negative_slope * grad_out[id];
                }

                kernel void mse_backward_kernel(
                    device const float* pred     [[buffer(0)]],
                    device const float* target   [[buffer(1)]],
                    device float* grad_pred      [[buffer(2)]],
                    constant float& scale        [[buffer(3)]],
                    constant uint& total_elems   [[buffer(4)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    grad_pred[id] = scale * (pred[id] - target[id]);
                }

                kernel void l1_loss_backward_kernel(
                    device const float* pred     [[buffer(0)]],
                    device const float* target   [[buffer(1)]],
                    device float* grad_pred      [[buffer(2)]],
                    constant float& scale        [[buffer(3)]],
                    constant uint& total_elems   [[buffer(4)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    float diff = pred[id] - target[id];
                    float sgn = diff > 0.0f ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
                    grad_pred[id] = scale * sgn;
                }

                kernel void causal_mask_kernel(
                    device const float* in       [[buffer(0)]],
                    device float* out            [[buffer(1)]],
                    constant uint& total_elems   [[buffer(2)]],
                    constant uint& seq_len       [[buffer(3)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    uint j = id % seq_len;
                    uint i = (id / seq_len) % seq_len;
                    out[id] = (j > i) ? -1e9f : in[id];
                }

                kernel void causal_mask_backward_kernel(
                    device const float* grad_out [[buffer(0)]],
                    device float* grad_in        [[buffer(1)]],
                    constant uint& total_elems   [[buffer(2)]],
                    constant uint& seq_len       [[buffer(3)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    uint j = id % seq_len;
                    uint i = (id / seq_len) % seq_len;
                    grad_in[id] = (j > i) ? 0.0f : grad_out[id];
                }

                kernel void sliding_window_mask_kernel(
                    device const float* in       [[buffer(0)]],
                    device float* out            [[buffer(1)]],
                    constant uint& total_elems   [[buffer(2)]],
                    constant uint& seq_len       [[buffer(3)]],
                    constant uint& window_size   [[buffer(4)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    uint j = id % seq_len;
                    uint i = (id / seq_len) % seq_len;
                    bool valid = (j <= i) && (i - j <= window_size);
                    out[id] = valid ? in[id] : -1e9f;
                }

                kernel void sliding_window_mask_backward_kernel(
                    device const float* grad_out [[buffer(0)]],
                    device float* grad_in        [[buffer(1)]],
                    constant uint& total_elems   [[buffer(2)]],
                    constant uint& seq_len       [[buffer(3)]],
                    constant uint& window_size   [[buffer(4)]],
                    uint id                      [[thread_position_in_grid]]
                ) {
                    if (id >= total_elems) return;
                    uint j = id % seq_len;
                    uint i = (id / seq_len) % seq_len;
                    bool valid = (j <= i) && (i - j <= window_size);
                    grad_in[id] = valid ? grad_out[id] : 0.0f;
                }
            )";

            NSError* error = nil;
            library = [device newLibraryWithSource:shaderSource options:nil error:&error];
            if (!library) {
                NSLog(@"[MetalBackend] Failed to compile Metal library: %@", error);
                return;
            }

            auto create_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> fn = [library newFunctionWithName:name];
                if (!fn) return nil;
                NSError* e = nil;
                id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&e];
                return pso;
            };

            pipeline_bias_gelu = create_pso(@"fused_add_bias_gelu_kernel");
            pipeline_bias_relu = create_pso(@"fused_add_bias_relu_kernel");
            pipeline_layernorm = create_pso(@"fused_layernorm_kernel");
            pipeline_adamw     = create_pso(@"fused_adamw_kernel");
            pipeline_attention = create_pso(@"tiled_flash_attention_kernel");

            pipeline_sigmoid        = create_pso(@"sigmoid_kernel");
            pipeline_sigmoid_bwd    = create_pso(@"sigmoid_backward_kernel");
            pipeline_silu           = create_pso(@"silu_kernel");
            pipeline_silu_bwd       = create_pso(@"silu_backward_kernel");
            pipeline_leaky_relu     = create_pso(@"leaky_relu_kernel");
            pipeline_leaky_relu_bwd = create_pso(@"leaky_relu_backward_kernel");
            pipeline_mse_bwd        = create_pso(@"mse_backward_kernel");
            pipeline_l1_loss_bwd    = create_pso(@"l1_loss_backward_kernel");

            pipeline_causal_mask            = create_pso(@"causal_mask_kernel");
            pipeline_causal_mask_bwd        = create_pso(@"causal_mask_backward_kernel");
            pipeline_sliding_window_mask     = create_pso(@"sliding_window_mask_kernel");
            pipeline_sliding_window_mask_bwd = create_pso(@"sliding_window_mask_backward_kernel");

            ready = true;
        }
    }

    bool is_available() const override { return ready; }

    std::string device_name() const override {
        if (!device) return "None";
        return [[device name] UTF8String];
    }

    void* allocate_shared_buffer(size_t bytes) override {
        if (!device) return nullptr;
        @autoreleasepool {
            id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!buf) return nullptr;
            void* ptr = [buf contents];
            std::lock_guard<std::mutex> lock(mtx);
            buffer_map[ptr] = buf;
            return ptr;
        }
    }

    void free_buffer(void* ptr, size_t bytes) override {
        (void)bytes;
        std::lock_guard<std::mutex> lock(mtx);
        auto it = buffer_map.find(ptr);
        if (it != buffer_map.end()) {
            buffer_map.erase(it);
        }
    }

    id<MTLBuffer> get_or_wrap_buffer(const void* ptr, size_t bytes) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = buffer_map.find(const_cast<void*>(ptr));
        if (it != buffer_map.end()) {
            return it->second;
        }
        // Zero-copy wrapper over host pointer
        id<MTLBuffer> wrapped = [device newBufferWithBytesNoCopy:const_cast<void*>(ptr)
                                                         length:bytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        buffer_map[const_cast<void*>(ptr)] = wrapped;
        return wrapped;
    }

    void matmul(
        const float* A, const float* B, float* C,
        int64_t M, int64_t K, int64_t N
    ) override {
        if (!ready) return;
        @autoreleasepool {
            id<MTLBuffer> bufA = get_or_wrap_buffer(A, M * K * sizeof(float));
            id<MTLBuffer> bufB = get_or_wrap_buffer(B, K * N * sizeof(float));
            id<MTLBuffer> bufC = get_or_wrap_buffer(C, M * N * sizeof(float));

            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                               columns:K
                                                                              rowBytes:K * sizeof(float)
                                                                              dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                               columns:N
                                                                              rowBytes:N * sizeof(float)
                                                                              dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                               columns:N
                                                                              rowBytes:N * sizeof(float)
                                                                              dataType:MPSDataTypeFloat32];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
            MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

            MPSMatrixMultiplication* mul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                             transposeLeft:NO
                                                                            transposeRight:NO
                                                                                resultRows:M
                                                                             resultColumns:N
                                                                           interiorColumns:K
                                                                                     alpha:1.0
                                                                                      beta:0.0];

            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            [mul encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void fused_add_bias_gelu(
        const float* in, const float* bias, float* out,
        size_t total_elems, size_t cols
    ) override {
        if (!ready || !pipeline_bias_gelu) return;
        @autoreleasepool {
            id<MTLBuffer> bufIn   = get_or_wrap_buffer(in, total_elems * sizeof(float));
            id<MTLBuffer> bufBias = get_or_wrap_buffer(bias, cols * sizeof(float));
            id<MTLBuffer> bufOut  = get_or_wrap_buffer(out, total_elems * sizeof(float));

            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_bias_gelu];
            [enc setBuffer:bufIn offset:0 atIndex:0];
            [enc setBuffer:bufBias offset:0 atIndex:1];
            [enc setBuffer:bufOut offset:0 atIndex:2];

            uint tot = static_cast<uint>(total_elems);
            uint c = static_cast<uint>(cols);
            [enc setBytes:&tot length:sizeof(tot) atIndex:3];
            [enc setBytes:&c length:sizeof(c) atIndex:4];

            MTLSize gridSize = MTLSizeMake(total_elems, 1, 1);
            NSUInteger w = pipeline_bias_gelu.threadExecutionWidth;
            MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void fused_add_bias_relu(
        const float* in, const float* bias, float* out,
        size_t total_elems, size_t cols
    ) override {
        if (!ready || !pipeline_bias_relu) return;
        @autoreleasepool {
            id<MTLBuffer> bufIn   = get_or_wrap_buffer(in, total_elems * sizeof(float));
            id<MTLBuffer> bufBias = get_or_wrap_buffer(bias, cols * sizeof(float));
            id<MTLBuffer> bufOut  = get_or_wrap_buffer(out, total_elems * sizeof(float));

            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_bias_relu];
            [enc setBuffer:bufIn offset:0 atIndex:0];
            [enc setBuffer:bufBias offset:0 atIndex:1];
            [enc setBuffer:bufOut offset:0 atIndex:2];

            uint tot = static_cast<uint>(total_elems);
            uint c = static_cast<uint>(cols);
            [enc setBytes:&tot length:sizeof(tot) atIndex:3];
            [enc setBytes:&c length:sizeof(c) atIndex:4];

            MTLSize gridSize = MTLSizeMake(total_elems, 1, 1);
            NSUInteger w = pipeline_bias_relu.threadExecutionWidth;
            MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void layernorm(
        const float* in, const float* gamma, const float* beta, float* out,
        size_t rows, size_t cols, float eps
    ) override {
        if (!ready || !pipeline_layernorm) return;
        @autoreleasepool {
            id<MTLBuffer> bufIn    = get_or_wrap_buffer(in, rows * cols * sizeof(float));
            id<MTLBuffer> bufGamma = gamma ? get_or_wrap_buffer(gamma, cols * sizeof(float)) : nil;
            id<MTLBuffer> bufBeta  = beta ? get_or_wrap_buffer(beta, cols * sizeof(float)) : nil;
            id<MTLBuffer> bufOut   = get_or_wrap_buffer(out, rows * cols * sizeof(float));

            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_layernorm];
            [enc setBuffer:bufIn offset:0 atIndex:0];
            [enc setBuffer:bufGamma offset:0 atIndex:1];
            [enc setBuffer:bufBeta offset:0 atIndex:2];
            [enc setBuffer:bufOut offset:0 atIndex:3];

            uint c = static_cast<uint>(cols);
            [enc setBytes:&c length:sizeof(c) atIndex:4];
            [enc setBytes:&eps length:sizeof(eps) atIndex:5];

            MTLSize threadgroups = MTLSizeMake(rows, 1, 1);
            MTLSize threadsPerGroup = MTLSizeMake(std::min<size_t>(cols, 256), 1, 1);
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
            [enc endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void adamw(
        float* p, const float* g, float* m, float* v,
        size_t total_params, float lr, float beta1, float beta2,
        float eps, float weight_decay, int step
    ) override {
        if (!ready || !pipeline_adamw) return;
        @autoreleasepool {
            id<MTLBuffer> bufP = get_or_wrap_buffer(p, total_params * sizeof(float));
            id<MTLBuffer> bufG = get_or_wrap_buffer(g, total_params * sizeof(float));
            id<MTLBuffer> bufM = get_or_wrap_buffer(m, total_params * sizeof(float));
            id<MTLBuffer> bufV = get_or_wrap_buffer(v, total_params * sizeof(float));

            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_adamw];
            [enc setBuffer:bufP offset:0 atIndex:0];
            [enc setBuffer:bufG offset:0 atIndex:1];
            [enc setBuffer:bufM offset:0 atIndex:2];
            [enc setBuffer:bufV offset:0 atIndex:3];

            float bc1 = 1.0f - std::pow(beta1, step);
            float bc2 = 1.0f - std::pow(beta2, step);
            uint tot = static_cast<uint>(total_params);

            [enc setBytes:&lr length:sizeof(lr) atIndex:4];
            [enc setBytes:&beta1 length:sizeof(beta1) atIndex:5];
            [enc setBytes:&beta2 length:sizeof(beta2) atIndex:6];
            [enc setBytes:&eps length:sizeof(eps) atIndex:7];
            [enc setBytes:&weight_decay length:sizeof(weight_decay) atIndex:8];
            [enc setBytes:&bc1 length:sizeof(bc1) atIndex:9];
            [enc setBytes:&bc2 length:sizeof(bc2) atIndex:10];
            [enc setBytes:&tot length:sizeof(tot) atIndex:11];

            MTLSize gridSize = MTLSizeMake(total_params, 1, 1);
            NSUInteger w = pipeline_adamw.threadExecutionWidth;
            MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void flash_attention(
        const float* Q, const float* K, const float* V, float* Out,
        size_t num_heads, size_t seq_len, size_t head_dim
    ) override {
        if (!ready || !pipeline_attention) return;
        @autoreleasepool {
            size_t total_elements = num_heads * seq_len * head_dim;
            id<MTLBuffer> bufQ   = get_or_wrap_buffer(Q, total_elements * sizeof(float));
            id<MTLBuffer> bufK   = get_or_wrap_buffer(K, total_elements * sizeof(float));
            id<MTLBuffer> bufV   = get_or_wrap_buffer(V, total_elements * sizeof(float));
            id<MTLBuffer> bufOut = get_or_wrap_buffer(Out, total_elements * sizeof(float));

            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_attention];
            [enc setBuffer:bufQ offset:0 atIndex:0];
            [enc setBuffer:bufK offset:0 atIndex:1];
            [enc setBuffer:bufV offset:0 atIndex:2];
            [enc setBuffer:bufOut offset:0 atIndex:3];

            uint s_len = static_cast<uint>(seq_len);
            uint h_dim = static_cast<uint>(head_dim);
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            [enc setBytes:&s_len length:sizeof(s_len) atIndex:4];
            [enc setBytes:&h_dim length:sizeof(h_dim) atIndex:5];
            [enc setBytes:&scale length:sizeof(scale) atIndex:6];

            MTLSize threadgroups = MTLSizeMake(seq_len, num_heads, 1);
            MTLSize threadsPerGroup = MTLSizeMake(std::min<size_t>(head_dim, 256), 1, 1);
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
            [enc endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void sigmoid_forward(const float* in, float* out, size_t size) override {
        if (!ready || !pipeline_sigmoid) return;
        @autoreleasepool {
            id<MTLBuffer> bufIn  = get_or_wrap_buffer(in, size * sizeof(float));
            id<MTLBuffer> bufOut = get_or_wrap_buffer(out, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_sigmoid];
            [enc setBuffer:bufIn offset:0 atIndex:0];
            [enc setBuffer:bufOut offset:0 atIndex:1];
            uint tot = static_cast<uint>(size);
            [enc setBytes:&tot length:sizeof(tot) atIndex:2];
            MTLSize gridSize = MTLSizeMake(size, 1, 1);
            NSUInteger w = pipeline_sigmoid.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void sigmoid_backward(const float* out, const float* grad_out, float* grad_in, size_t size) override {
        if (!ready || !pipeline_sigmoid_bwd) return;
        @autoreleasepool {
            id<MTLBuffer> bufOut  = get_or_wrap_buffer(out, size * sizeof(float));
            id<MTLBuffer> bufGOut = get_or_wrap_buffer(grad_out, size * sizeof(float));
            id<MTLBuffer> bufGIn  = get_or_wrap_buffer(grad_in, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_sigmoid_bwd];
            [enc setBuffer:bufOut offset:0 atIndex:0];
            [enc setBuffer:bufGOut offset:0 atIndex:1];
            [enc setBuffer:bufGIn offset:0 atIndex:2];
            uint tot = static_cast<uint>(size);
            [enc setBytes:&tot length:sizeof(tot) atIndex:3];
            MTLSize gridSize = MTLSizeMake(size, 1, 1);
            NSUInteger w = pipeline_sigmoid_bwd.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void silu_forward(const float* in, float* out, size_t size) override {
        if (!ready || !pipeline_silu) return;
        @autoreleasepool {
            id<MTLBuffer> bufIn  = get_or_wrap_buffer(in, size * sizeof(float));
            id<MTLBuffer> bufOut = get_or_wrap_buffer(out, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_silu];
            [enc setBuffer:bufIn offset:0 atIndex:0];
            [enc setBuffer:bufOut offset:0 atIndex:1];
            uint tot = static_cast<uint>(size);
            [enc setBytes:&tot length:sizeof(tot) atIndex:2];
            MTLSize gridSize = MTLSizeMake(size, 1, 1);
            NSUInteger w = pipeline_silu.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void silu_backward(const float* in, const float* grad_out, float* grad_in, size_t size) override {
        if (!ready || !pipeline_silu_bwd) return;
        @autoreleasepool {
            id<MTLBuffer> bufIn   = get_or_wrap_buffer(in, size * sizeof(float));
            id<MTLBuffer> bufGOut = get_or_wrap_buffer(grad_out, size * sizeof(float));
            id<MTLBuffer> bufGIn  = get_or_wrap_buffer(grad_in, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_silu_bwd];
            [enc setBuffer:bufIn offset:0 atIndex:0];
            [enc setBuffer:bufGOut offset:0 atIndex:1];
            [enc setBuffer:bufGIn offset:0 atIndex:2];
            uint tot = static_cast<uint>(size);
            [enc setBytes:&tot length:sizeof(tot) atIndex:3];
            MTLSize gridSize = MTLSizeMake(size, 1, 1);
            NSUInteger w = pipeline_silu_bwd.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void leaky_relu_forward(const float* in, float* out, size_t size, float negative_slope) override {
        if (!ready || !pipeline_leaky_relu) return;
        @autoreleasepool {
            id<MTLBuffer> bufIn  = get_or_wrap_buffer(in, size * sizeof(float));
            id<MTLBuffer> bufOut = get_or_wrap_buffer(out, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_leaky_relu];
            [enc setBuffer:bufIn offset:0 atIndex:0];
            [enc setBuffer:bufOut offset:0 atIndex:1];
            [enc setBytes:&negative_slope length:sizeof(negative_slope) atIndex:2];
            uint tot = static_cast<uint>(size);
            [enc setBytes:&tot length:sizeof(tot) atIndex:3];
            MTLSize gridSize = MTLSizeMake(size, 1, 1);
            NSUInteger w = pipeline_leaky_relu.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void leaky_relu_backward(const float* in, const float* grad_out, float* grad_in, size_t size, float negative_slope) override {
        if (!ready || !pipeline_leaky_relu_bwd) return;
        @autoreleasepool {
            id<MTLBuffer> bufIn   = get_or_wrap_buffer(in, size * sizeof(float));
            id<MTLBuffer> bufGOut = get_or_wrap_buffer(grad_out, size * sizeof(float));
            id<MTLBuffer> bufGIn  = get_or_wrap_buffer(grad_in, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_leaky_relu_bwd];
            [enc setBuffer:bufIn offset:0 atIndex:0];
            [enc setBuffer:bufGOut offset:0 atIndex:1];
            [enc setBuffer:bufGIn offset:0 atIndex:2];
            [enc setBytes:&negative_slope length:sizeof(negative_slope) atIndex:3];
            uint tot = static_cast<uint>(size);
            [enc setBytes:&tot length:sizeof(tot) atIndex:4];
            MTLSize gridSize = MTLSizeMake(size, 1, 1);
            NSUInteger w = pipeline_leaky_relu_bwd.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void mse_backward(const float* pred, const float* target, float* grad_pred, size_t size, float scale) override {
        if (!ready || !pipeline_mse_bwd) return;
        @autoreleasepool {
            id<MTLBuffer> bufP = get_or_wrap_buffer(pred, size * sizeof(float));
            id<MTLBuffer> bufT = get_or_wrap_buffer(target, size * sizeof(float));
            id<MTLBuffer> bufG = get_or_wrap_buffer(grad_pred, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_mse_bwd];
            [enc setBuffer:bufP offset:0 atIndex:0];
            [enc setBuffer:bufT offset:0 atIndex:1];
            [enc setBuffer:bufG offset:0 atIndex:2];
            [enc setBytes:&scale length:sizeof(scale) atIndex:3];
            uint tot = static_cast<uint>(size);
            [enc setBytes:&tot length:sizeof(tot) atIndex:4];
            MTLSize gridSize = MTLSizeMake(size, 1, 1);
            NSUInteger w = pipeline_mse_bwd.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void l1_loss_backward(const float* pred, const float* target, float* grad_pred, size_t size, float scale) override {
        if (!ready || !pipeline_l1_loss_bwd) return;
        @autoreleasepool {
            id<MTLBuffer> bufP = get_or_wrap_buffer(pred, size * sizeof(float));
            id<MTLBuffer> bufT = get_or_wrap_buffer(target, size * sizeof(float));
            id<MTLBuffer> bufG = get_or_wrap_buffer(grad_pred, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_l1_loss_bwd];
            [enc setBuffer:bufP offset:0 atIndex:0];
            [enc setBuffer:bufT offset:0 atIndex:1];
            [enc setBuffer:bufG offset:0 atIndex:2];
            [enc setBytes:&scale length:sizeof(scale) atIndex:3];
            uint tot = static_cast<uint>(size);
            [enc setBytes:&tot length:sizeof(tot) atIndex:4];
            MTLSize gridSize = MTLSizeMake(size, 1, 1);
            NSUInteger w = pipeline_l1_loss_bwd.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void causal_mask(const float* in, float* out, size_t batch, size_t seq_len) override {
        if (!ready || !pipeline_causal_mask) return;
        size_t total = batch * seq_len * seq_len;
        @autoreleasepool {
            id<MTLBuffer> bufIn  = get_or_wrap_buffer(in, total * sizeof(float));
            id<MTLBuffer> bufOut = get_or_wrap_buffer(out, total * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_causal_mask];
            [enc setBuffer:bufIn offset:0 atIndex:0];
            [enc setBuffer:bufOut offset:0 atIndex:1];
            uint tot = static_cast<uint>(total);
            uint slen = static_cast<uint>(seq_len);
            [enc setBytes:&tot length:sizeof(tot) atIndex:2];
            [enc setBytes:&slen length:sizeof(slen) atIndex:3];
            MTLSize gridSize = MTLSizeMake(total, 1, 1);
            NSUInteger w = pipeline_causal_mask.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void causal_mask_backward(const float* grad_out, float* grad_in, size_t batch, size_t seq_len) override {
        if (!ready || !pipeline_causal_mask_bwd) return;
        size_t total = batch * seq_len * seq_len;
        @autoreleasepool {
            id<MTLBuffer> bufGOut = get_or_wrap_buffer(grad_out, total * sizeof(float));
            id<MTLBuffer> bufGIn  = get_or_wrap_buffer(grad_in, total * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_causal_mask_bwd];
            [enc setBuffer:bufGOut offset:0 atIndex:0];
            [enc setBuffer:bufGIn offset:0 atIndex:1];
            uint tot = static_cast<uint>(total);
            uint slen = static_cast<uint>(seq_len);
            [enc setBytes:&tot length:sizeof(tot) atIndex:2];
            [enc setBytes:&slen length:sizeof(slen) atIndex:3];
            MTLSize gridSize = MTLSizeMake(total, 1, 1);
            NSUInteger w = pipeline_causal_mask_bwd.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void sliding_window_mask(const float* in, float* out, size_t batch, size_t seq_len, size_t window_size) override {
        if (!ready || !pipeline_sliding_window_mask) return;
        size_t total = batch * seq_len * seq_len;
        @autoreleasepool {
            id<MTLBuffer> bufIn  = get_or_wrap_buffer(in, total * sizeof(float));
            id<MTLBuffer> bufOut = get_or_wrap_buffer(out, total * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_sliding_window_mask];
            [enc setBuffer:bufIn offset:0 atIndex:0];
            [enc setBuffer:bufOut offset:0 atIndex:1];
            uint tot = static_cast<uint>(total);
            uint slen = static_cast<uint>(seq_len);
            uint win = static_cast<uint>(window_size);
            [enc setBytes:&tot length:sizeof(tot) atIndex:2];
            [enc setBytes:&slen length:sizeof(slen) atIndex:3];
            [enc setBytes:&win length:sizeof(win) atIndex:4];
            MTLSize gridSize = MTLSizeMake(total, 1, 1);
            NSUInteger w = pipeline_sliding_window_mask.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void sliding_window_mask_backward(const float* grad_out, float* grad_in, size_t batch, size_t seq_len, size_t window_size) override {
        if (!ready || !pipeline_sliding_window_mask_bwd) return;
        size_t total = batch * seq_len * seq_len;
        @autoreleasepool {
            id<MTLBuffer> bufGOut = get_or_wrap_buffer(grad_out, total * sizeof(float));
            id<MTLBuffer> bufGIn  = get_or_wrap_buffer(grad_in, total * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_sliding_window_mask_bwd];
            [enc setBuffer:bufGOut offset:0 atIndex:0];
            [enc setBuffer:bufGIn offset:0 atIndex:1];
            uint tot = static_cast<uint>(total);
            uint slen = static_cast<uint>(seq_len);
            uint win = static_cast<uint>(window_size);
            [enc setBytes:&tot length:sizeof(tot) atIndex:2];
            [enc setBytes:&slen length:sizeof(slen) atIndex:3];
            [enc setBytes:&win length:sizeof(win) atIndex:4];
            MTLSize gridSize = MTLSizeMake(total, 1, 1);
            NSUInteger w = pipeline_sliding_window_mask_bwd.threadExecutionWidth;
            [enc dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
    }

    void synchronize() override {
        // Managed synchronously per dispatch queue command buffer
    }
};

MetalBackend& MetalBackend::get() {
    static MetalBackendImpl instance;
    return instance;
}

#else

// Non-Apple fallback stub
class DummyMetalBackend : public MetalBackend {
public:
    bool is_available() const override { return false; }
    std::string device_name() const override { return "Metal unavailable on this OS"; }
    void* allocate_shared_buffer(size_t) override { return nullptr; }
    void free_buffer(void*, size_t) override {}
    void matmul(const float*, const float*, float*, int64_t, int64_t, int64_t) override {}
    void fused_add_bias_gelu(const float*, const float*, float*, size_t, size_t) override {}
    void fused_add_bias_relu(const float*, const float*, float*, size_t, size_t) override {}
    void layernorm(const float*, const float*, const float*, float*, size_t, size_t, float) override {}
    void adamw(float*, const float*, float*, float*, size_t, float, float, float, float, float, int) override {}
    void sigmoid_forward(const float*, float*, size_t) override {}
    void sigmoid_backward(const float*, const float*, float*, size_t) override {}
    void silu_forward(const float*, float*, size_t) override {}
    void silu_backward(const float*, const float*, float*, size_t) override {}
    void leaky_relu_forward(const float*, float*, size_t, float) override {}
    void leaky_relu_backward(const float*, const float*, float*, size_t, float) override {}
    void mse_backward(const float*, const float*, float*, size_t, float) override {}
    void l1_loss_backward(const float*, const float*, float*, size_t, float) override {}
    void flash_attention(const float*, const float*, const float*, float*, size_t, size_t, size_t) override {}
    void causal_mask(const float*, float*, size_t, size_t) override {}
    void causal_mask_backward(const float*, float*, size_t, size_t) override {}
    void sliding_window_mask(const float*, float*, size_t, size_t, size_t) override {}
    void sliding_window_mask_backward(const float*, float*, size_t, size_t, size_t) override {}
    void synchronize() override {}
};

MetalBackend& MetalBackend::get() {
    static DummyMetalBackend instance;
    return instance;
}

#endif
