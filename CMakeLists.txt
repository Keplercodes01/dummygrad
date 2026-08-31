cmake_minimum_required(VERSION 3.14)
project(dummygrad)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compile everything into a static library for the engine
add_library(dummygrad_engine STATIC
    src/storage.cpp
    src/utils.cpp
    src/node.cpp
    src/autograd.cpp
    src/tensor.cpp
    src/allocator.cpp
    src/ops.cpp
    src/cpu_ops.cpp
)

target_include_directories(dummygrad_engine PUBLIC src)

# Optional: Add an executable for testing
# add_executable(dummygrad_test src/main.cpp)
# target_link_libraries(dummygrad_test dummygrad_engine)

find_library(ACCELERATE_FRAMEWORK Accelerate)
target_link_libraries(dummygrad_engine ${ACCELERATE_FRAMEWORK})
