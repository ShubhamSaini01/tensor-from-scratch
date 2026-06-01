cmake_minimum_required(VERSION 3.16)
project(tensor-from-scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Core library ──
add_library(tfs_core
    src/tensor.cpp
)
target_include_directories(tfs_core PUBLIC include)

# Warnings — catch bugs early
target_compile_options(tfs_core PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
)

# ── Stage 1 tests ──
add_executable(test_stage1 tests/test_stage1.cpp)
target_link_libraries(test_stage1 PRIVATE tfs_core)

# ── Future stages (uncomment as you go) ──
# add_executable(test_stage2 tests/test_stage2.cpp)
# target_link_libraries(test_stage2 PRIVATE tfs_core)

# Stage 4+: CUDA
# enable_language(CUDA)
# add_library(tfs_cuda src/tensor_cuda.cu)
# target_link_libraries(tfs_cuda PUBLIC tfs_core)
