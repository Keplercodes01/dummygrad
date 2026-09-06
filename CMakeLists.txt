cmake_minimum_required(VERSION 3.18)
project(dummygrad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source files for the core engine
set(DUMMYGRAD_SOURCES
    src/storage.cpp
    src/utils.cpp
    src/node.cpp
    src/autograd.cpp
    src/tensor.cpp
    src/allocator.cpp
    src/ops.cpp
    src/cpu_ops.cpp
)

# Detect and configure CUDA for GPU execution (e.g. Google Colab / Linux GPU)
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
    list(APPEND DUMMYGRAD_SOURCES src/cuda/cuda_ops.cu)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}. Building with GPU support.")
else()
    message(STATUS "CUDA not found. Building CPU-only.")
endif()

# Build engine static library
add_library(dummygrad_engine STATIC ${DUMMYGRAD_SOURCES})
target_include_directories(dummygrad_engine PUBLIC src)

# Optional BLAS (OpenBLAS / oneMKL / MKL / Netlib) for fast CPU linear algebra
find_package(BLAS QUIET)
if(BLAS_FOUND)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    target_link_libraries(dummygrad_engine PRIVATE ${BLAS_LIBRARIES})
    target_compile_definitions(dummygrad_engine PUBLIC USE_BLAS=1)
endif()

# Link CUDA libraries when CUDA is enabled
if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(dummygrad_engine PRIVATE CUDA::cudart CUDA::cublas)
    target_compile_definitions(dummygrad_engine PUBLIC USE_CUDA=1)
endif()

# Optional: Add an executable for testing
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(dummygrad_test src/main.cpp)
    target_link_libraries(dummygrad_test PRIVATE dummygrad_engine)
endif()
