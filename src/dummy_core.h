#pragma once

// 1. Core Engine Primitives
#include "core/types.h"
#include "core/storage.h"
#include "core/allocator.h"
#include "core/tensor.h"
#include "core/utils.h"
#include "core/init.h"
#include "core/dummy_ptr.h"
#include "core/dummy_wrappers.h"

// 2. Autograd DAG & Engine
#include "autograd/node.h"
#include "autograd/autograd.h"

// 3. Math, Functional Operations & Masking
#include "ops/ops.h"
#include "ops/functional.h"
#include "ops/cool_ops.h"
#include "ops/masking.h"

// 4. Hardware Backends
#include "backends/metal/metal_backend.h"
#include "backends/tpu/pjrt_client.h"
#include "backends/tpu/hlo_builder.h"
#include "backends/tpu/tpu_runner.h"

// 5. Neural Network Layers & Optimizers
#include "nn/linear.h"
#include "nn/embedding.h"
#include "nn/layernorm.h"
#include "nn/modern_layers.h"
#include "nn/attention.h"
#include "nn/conv1d.h"
#include "nn/conv2d.h"
#include "nn/maxpool2d.h"
#include "nn/optimizers.h"

// 6. Reference Architectures
#include "models/transformer.h"
#include "models/mars.h"

// Multi-GPU Distributed Engine & Checkpoint Serialization
#include "multi_gpu.h"
#include "serialization.h"
