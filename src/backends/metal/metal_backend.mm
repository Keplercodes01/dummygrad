#include "metal_backend.h"

#if defined(__APPLE__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstdlib>
#include <map>
#include <unistd.h>
#include <mutex>
#include <cmath>
#include <iostream>

class MetalBackendImpl : public MetalBackend {
private:
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> library = nil;

    id<MTLComputePipelineState> pipeline_adamw = nil;

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

            pipeline_adamw     = create_pso(@"fused_adamw_kernel");

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

    // Buffers this backend allocated, keyed by their contents' address. A tensor's data pointer may
    // point inside one (a view with an offset), so lookups find the buffer containing it.
    struct OwnedBuffer {
        id<MTLBuffer> buffer;
        size_t length;
    };
    std::map<uintptr_t, OwnedBuffer> owned;

    // A kernel argument: a buffer and the byte offset of the data inside it
    struct BufferRef {
        id<MTLBuffer> buf;
        NSUInteger offset;
    };

    void* allocate_shared_buffer(size_t bytes) override {
        if (!device || bytes == 0) return nullptr;
        @autoreleasepool {
            id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!buf) return nullptr;
            void* ptr = [buf contents];
            std::lock_guard<std::mutex> lock(mtx);
            owned[reinterpret_cast<uintptr_t>(ptr)] = OwnedBuffer{buf, bytes};
            return ptr;
        }
    }

    void free_buffer(void* ptr, size_t bytes) override {
        (void)bytes;
        std::lock_guard<std::mutex> lock(mtx);
        auto it = owned.find(reinterpret_cast<uintptr_t>(ptr));
        if (it == owned.end()) {
            std::free(ptr);  // allocate_shared_buffer failed and the allocator fell back to malloc
            return;
        }
#if !__has_feature(objc_arc)
        [it->second.buffer release];  // newBufferWithLength returned it retained
#endif
        owned.erase(it);
    }

    // The buffer holding [ptr, ptr + bytes): one of ours (with the view's offset), or, for host memory
    // from elsewhere, a no-copy wrapper over the page-aligned range around it (released with the
    // enclosing autorelease pool)
    BufferRef get_or_wrap_buffer(const void* ptr, size_t bytes) {
        uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = owned.upper_bound(p);
            if (it != owned.begin()) {
                --it;
                if (p + bytes <= it->first + it->second.length) return BufferRef{it->second.buffer, (NSUInteger)(p - it->first)};
            }
        }
        uintptr_t page = (uintptr_t)getpagesize();
        uintptr_t base = p & ~(page - 1);
        size_t length = ((p + bytes - base) + page - 1) & ~(page - 1);
        id<MTLBuffer> wrapped = [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(base)
                                                          length:length
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
#if !__has_feature(objc_arc)
        [wrapped autorelease];
#endif
        return BufferRef{wrapped, (NSUInteger)(p - base)};
    }

    void adamw(
        float* p, const float* g, float* m, float* v,
        size_t total_params, float lr, float beta1, float beta2,
        float eps, float weight_decay, int step
    ) override {
        if (!ready || !pipeline_adamw) return;
        @autoreleasepool {
            BufferRef bufP = get_or_wrap_buffer(p, total_params * sizeof(float));
            BufferRef bufG = get_or_wrap_buffer(g, total_params * sizeof(float));
            BufferRef bufM = get_or_wrap_buffer(m, total_params * sizeof(float));
            BufferRef bufV = get_or_wrap_buffer(v, total_params * sizeof(float));

            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_adamw];
            [enc setBuffer:bufP.buf offset:bufP.offset atIndex:0];
            [enc setBuffer:bufG.buf offset:bufG.offset atIndex:1];
            [enc setBuffer:bufM.buf offset:bufM.offset atIndex:2];
            [enc setBuffer:bufV.buf offset:bufV.offset atIndex:3];

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

    void sigmoid_forward(const float* in, float* out, size_t size) override {
        if (!ready || !pipeline_sigmoid) return;
        @autoreleasepool {
            BufferRef bufIn = get_or_wrap_buffer(in, size * sizeof(float));
            BufferRef bufOut = get_or_wrap_buffer(out, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_sigmoid];
            [enc setBuffer:bufIn.buf offset:bufIn.offset atIndex:0];
            [enc setBuffer:bufOut.buf offset:bufOut.offset atIndex:1];
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
            BufferRef bufOut = get_or_wrap_buffer(out, size * sizeof(float));
            BufferRef bufGOut = get_or_wrap_buffer(grad_out, size * sizeof(float));
            BufferRef bufGIn = get_or_wrap_buffer(grad_in, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_sigmoid_bwd];
            [enc setBuffer:bufOut.buf offset:bufOut.offset atIndex:0];
            [enc setBuffer:bufGOut.buf offset:bufGOut.offset atIndex:1];
            [enc setBuffer:bufGIn.buf offset:bufGIn.offset atIndex:2];
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
            BufferRef bufIn = get_or_wrap_buffer(in, size * sizeof(float));
            BufferRef bufOut = get_or_wrap_buffer(out, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_silu];
            [enc setBuffer:bufIn.buf offset:bufIn.offset atIndex:0];
            [enc setBuffer:bufOut.buf offset:bufOut.offset atIndex:1];
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
            BufferRef bufIn = get_or_wrap_buffer(in, size * sizeof(float));
            BufferRef bufGOut = get_or_wrap_buffer(grad_out, size * sizeof(float));
            BufferRef bufGIn = get_or_wrap_buffer(grad_in, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_silu_bwd];
            [enc setBuffer:bufIn.buf offset:bufIn.offset atIndex:0];
            [enc setBuffer:bufGOut.buf offset:bufGOut.offset atIndex:1];
            [enc setBuffer:bufGIn.buf offset:bufGIn.offset atIndex:2];
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
            BufferRef bufIn = get_or_wrap_buffer(in, size * sizeof(float));
            BufferRef bufOut = get_or_wrap_buffer(out, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_leaky_relu];
            [enc setBuffer:bufIn.buf offset:bufIn.offset atIndex:0];
            [enc setBuffer:bufOut.buf offset:bufOut.offset atIndex:1];
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
            BufferRef bufIn = get_or_wrap_buffer(in, size * sizeof(float));
            BufferRef bufGOut = get_or_wrap_buffer(grad_out, size * sizeof(float));
            BufferRef bufGIn = get_or_wrap_buffer(grad_in, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_leaky_relu_bwd];
            [enc setBuffer:bufIn.buf offset:bufIn.offset atIndex:0];
            [enc setBuffer:bufGOut.buf offset:bufGOut.offset atIndex:1];
            [enc setBuffer:bufGIn.buf offset:bufGIn.offset atIndex:2];
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
            BufferRef bufP = get_or_wrap_buffer(pred, size * sizeof(float));
            BufferRef bufT = get_or_wrap_buffer(target, size * sizeof(float));
            BufferRef bufG = get_or_wrap_buffer(grad_pred, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_mse_bwd];
            [enc setBuffer:bufP.buf offset:bufP.offset atIndex:0];
            [enc setBuffer:bufT.buf offset:bufT.offset atIndex:1];
            [enc setBuffer:bufG.buf offset:bufG.offset atIndex:2];
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
            BufferRef bufP = get_or_wrap_buffer(pred, size * sizeof(float));
            BufferRef bufT = get_or_wrap_buffer(target, size * sizeof(float));
            BufferRef bufG = get_or_wrap_buffer(grad_pred, size * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_l1_loss_bwd];
            [enc setBuffer:bufP.buf offset:bufP.offset atIndex:0];
            [enc setBuffer:bufT.buf offset:bufT.offset atIndex:1];
            [enc setBuffer:bufG.buf offset:bufG.offset atIndex:2];
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
            BufferRef bufIn = get_or_wrap_buffer(in, total * sizeof(float));
            BufferRef bufOut = get_or_wrap_buffer(out, total * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_causal_mask];
            [enc setBuffer:bufIn.buf offset:bufIn.offset atIndex:0];
            [enc setBuffer:bufOut.buf offset:bufOut.offset atIndex:1];
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
            BufferRef bufGOut = get_or_wrap_buffer(grad_out, total * sizeof(float));
            BufferRef bufGIn = get_or_wrap_buffer(grad_in, total * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_causal_mask_bwd];
            [enc setBuffer:bufGOut.buf offset:bufGOut.offset atIndex:0];
            [enc setBuffer:bufGIn.buf offset:bufGIn.offset atIndex:1];
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
            BufferRef bufIn = get_or_wrap_buffer(in, total * sizeof(float));
            BufferRef bufOut = get_or_wrap_buffer(out, total * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_sliding_window_mask];
            [enc setBuffer:bufIn.buf offset:bufIn.offset atIndex:0];
            [enc setBuffer:bufOut.buf offset:bufOut.offset atIndex:1];
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
            BufferRef bufGOut = get_or_wrap_buffer(grad_out, total * sizeof(float));
            BufferRef bufGIn = get_or_wrap_buffer(grad_in, total * sizeof(float));
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_sliding_window_mask_bwd];
            [enc setBuffer:bufGOut.buf offset:bufGOut.offset atIndex:0];
            [enc setBuffer:bufGIn.buf offset:bufGIn.offset atIndex:1];
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
    void adamw(float*, const float*, float*, float*, size_t, float, float, float, float, float, int) override {}
    void sigmoid_forward(const float*, float*, size_t) override {}
    void sigmoid_backward(const float*, const float*, float*, size_t) override {}
    void silu_forward(const float*, float*, size_t) override {}
    void silu_backward(const float*, const float*, float*, size_t) override {}
    void leaky_relu_forward(const float*, float*, size_t, float) override {}
    void leaky_relu_backward(const float*, const float*, float*, size_t, float) override {}
    void mse_backward(const float*, const float*, float*, size_t, float) override {}
    void l1_loss_backward(const float*, const float*, float*, size_t, float) override {}
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
