# A minimal downstream project, identical to the README's FetchContent instructions.
# CI points FETCHCONTENT_SOURCE_DIR_DUMMYGRAD at the checkout under test, so this
# verifies that dummygrad works when consumed by another CMake project.
cmake_minimum_required(VERSION 3.18)
project(my_app LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(dummygrad
    GIT_REPOSITORY https://github.com/Keplercodes01/dummygrad.git
    GIT_TAG        v0.4.0)
FetchContent_MakeAvailable(dummygrad)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE dummygrad::dummygrad)
