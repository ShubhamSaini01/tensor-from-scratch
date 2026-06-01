# tensor.cpp

A tensor class built from scratch in C++ and CUDA — from raw memory buffers to a paged KV cache allocator.

## Why

To understand what frameworks like PyTorch do under the hood, and to have a concrete artifact for GPU/inference systems interviews.


## Structure

```
include/     # headers (.hpp, .cuh)
src/         # implementation (.cpp, .cu)
tests/       # one test file per stage
benchmarks/  # Nsight-measurable benchmarks (stage 5+)
```

## Build

```
cmake -B build && cmake --build build
```

## Hardware

Developed on RTX 5070 (sm_120, Blackwell).