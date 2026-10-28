cmake_minimum_required(VERSION 3.18)
project(dummygrad VERSION 0.8.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Tests and examples are built by default only when dummygrad is the top-level project,
# not when another project pulls it in with add_subdirectory / FetchContent.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(DUMMYGRAD_IS_TOP_LEVEL ON)
else()
    set(DUMMYGRAD_IS_TOP_LEVEL OFF)
endif()

option(ENABLE_CUDA "Enable NVIDIA CUDA GPU acceleration" ON)
option(DUMMYGRAD_BUILD_TESTS "Build the gradient-check test suite (test_engine)" ${DUMMYGRAD_IS_TOP_LEVEL})
option(DUMMYGRAD_BUILD_EXAMPLES "Build the examples (quickstart, custom_op, train_lm, train_colab)" ${DUMMYGRAD_IS_TOP_LEVEL})
option(DUMMYGRAD_BUILD_BENCHMARKS "Build the training-step benchmark (dg_bench)" ${DUMMYGRAD_IS_TOP_LEVEL})

# Source files for the core engine
set(DUMMYGRAD_SOURCES
    src/core/storage.cpp
    src/core/utils.cpp
    src/autograd/node.cpp
    src/autograd/autograd.cpp
    src/core/tensor.cpp
    src/core/allocator.cpp
    src/core/profiler.cpp
    src/ops/ops.cpp
    src/backends/cpu/cpu_ops.cpp
    src/backends/tpu/pjrt_client.cpp
)

# Detect and configure CUDA for GPU execution
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # Compile for the GPU in this machine when CMake can detect it (Colab, cloud boxes),
        # otherwise for the common datacenter/consumer generations (T4, A100, RTX 30xx).
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
                set(CMAKE_CUDA_ARCHITECTURES native)
            else()
                set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
            endif()
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
        list(APPEND DUMMYGRAD_SOURCES src/backends/cuda/cuda_ops.cu src/backends/cuda/cuda_core_ops.cu src/backends/cuda/cuda_attention.cu)
        message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} (archs: ${CMAKE_CUDA_ARCHITECTURES}). Building with GPU support.")
    else()
        message(STATUS "CUDA compiler not found. Building CPU-only.")
    endif()
else()
    message(STATUS "CUDA disabled by user (ENABLE_CUDA=OFF). Building CPU-only.")
endif()

# Build engine static library
add_library(dummygrad_engine STATIC ${DUMMYGRAD_SOURCES})
add_library(dummygrad::dummygrad ALIAS dummygrad_engine)
target_include_directories(dummygrad_engine PUBLIC
    src
    src/core
    src/autograd
    src/ops
    src/nn
    src/backends
    src/models
)
target_link_libraries(dummygrad_engine PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(dummygrad_engine PUBLIC DUMMYGRAD_VERSION="${PROJECT_VERSION}")

# The autograd engine runs backward on a worker thread pool
# Imported targets from find_package are only visible in the directory that created them.
# These are part of dummygrad_engine's public link interface, so when dummygrad is pulled in
# with add_subdirectory / FetchContent they must be global or consumers can't resolve them.
if(NOT TARGET Threads::Threads)
    find_package(Threads REQUIRED)
    set_target_properties(Threads::Threads PROPERTIES IMPORTED_GLOBAL TRUE)
endif()
target_link_libraries(dummygrad_engine PUBLIC Threads::Threads)

# Optional BLAS (OpenBLAS / oneMKL / MKL / Netlib) for fast CPU linear algebra
find_package(BLAS QUIET)
if(BLAS_FOUND)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    target_link_libraries(dummygrad_engine PRIVATE ${BLAS_LIBRARIES})
    target_compile_definitions(dummygrad_engine PUBLIC USE_BLAS=1)
endif()

# Link CUDA libraries when CUDA is enabled. PUBLIC because the public headers
# (optimizers.h, multi_gpu.h, ...) call the CUDA runtime when USE_CUDA is defined.
if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    if(NOT TARGET CUDA::cudart)
        find_package(CUDAToolkit REQUIRED)
        set_target_properties(CUDA::cudart CUDA::cublas CUDA::cublasLt PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
    target_link_libraries(dummygrad_engine PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
    target_compile_definitions(dummygrad_engine PUBLIC USE_CUDA=1)
endif()

# Configure Apple Silicon Metal and MetalPerformanceShaders on macOS
if(APPLE)
    enable_language(OBJCXX)
    target_sources(dummygrad_engine PRIVATE src/backends/metal/metal_backend.mm)
    find_library(METAL_LIB Metal REQUIRED)
    find_library(MPS_LIB MetalPerformanceShaders REQUIRED)
    find_library(FOUNDATION_LIB Foundation REQUIRED)
    target_link_libraries(dummygrad_engine PRIVATE ${METAL_LIB} ${MPS_LIB} ${FOUNDATION_LIB})
    target_compile_definitions(dummygrad_engine PUBLIC USE_METAL=1)
    message(STATUS "Apple Silicon Metal & MPS configured.")
endif()

# Optional: Add executables for testing
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(dummygrad_test src/main.cpp)
    target_link_libraries(dummygrad_test PRIVATE dummygrad_engine)
endif()

if(DUMMYGRAD_BUILD_TESTS OR DUMMYGRAD_BUILD_EXAMPLES)
    enable_testing()
endif()

if(DUMMYGRAD_BUILD_EXAMPLES)
    add_executable(quickstart examples/quickstart.cpp)
    target_link_libraries(quickstart PRIVATE dummygrad_engine)
    add_test(NAME example_quickstart COMMAND quickstart)

    add_executable(custom_op examples/custom_op.cpp)
    target_link_libraries(custom_op PRIVATE dummygrad_engine)
    add_test(NAME example_custom_op COMMAND custom_op)

    add_executable(train_lm examples/train_lm.cpp)
    target_link_libraries(train_lm PRIVATE dummygrad_engine)
    add_test(NAME example_train_lm COMMAND train_lm)

    add_executable(train_colab src/train_colab.cpp)
    target_link_libraries(train_colab PRIVATE dummygrad_engine)
endif()

# Training-step benchmark; bench/bench_torch.py is the PyTorch reference
if(DUMMYGRAD_BUILD_BENCHMARKS)
    add_executable(dg_bench bench/bench.cpp)
    target_link_libraries(dg_bench PRIVATE dummygrad_engine)
endif()

# Gradient-check test suite: `ctest --output-on-failure` or run ./test_engine directly
if(DUMMYGRAD_BUILD_TESTS)
    add_executable(test_engine tests/test_engine.cpp)
    target_link_libraries(test_engine PRIVATE dummygrad_engine)
    add_test(NAME test_engine COMMAND test_engine)
endif()
