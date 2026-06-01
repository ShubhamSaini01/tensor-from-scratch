// Stage 1 tests — every green assert is a dopamine hit.
// Run: ./tests/test_stage1
// Goal: all 8 tests pass → you understand memory layout cold.

#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace tfs;

#define TEST(name)                                              \
    static void test_##name();                                  \
    struct Register_##name {                                    \
        Register_##name() { tests().push_back({#name, test_##name}); } \
    } reg_##name;                                               \
    static void test_##name()

static std::vector<std::pair<const char*, void(*)()>>& tests() {
    static std::vector<std::pair<const char*, void(*)()>> t;
    return t;
}

// ─── Test 1: Construction + shape ───────────────────────────────────
// "What's the shape of a tensor allocated with [2,3,4]?"
TEST(construction) {
    Tensor t({2, 3, 4});
    assert(t.ndim() == 3);
    assert(t.size(0) == 2);
    assert(t.size(1) == 3);
    assert(t.size(2) == 4);
    assert(t.numel() == 24);
}

// ─── Test 2: Strides are row-major ──────────────────────────────────
// "Given shape [2,3,4], what are the strides?"
// Answer: [12, 4, 1] — this is the bread-and-butter question.
TEST(strides_row_major) {
    Tensor t({2, 3, 4});
    auto s = t.strides();
    assert(s[0] == 12); // skip 3*4 elements to next "batch"
    assert(s[1] == 4);  // skip 4 elements to next row
    assert(s[2] == 1);  // consecutive in memory
}

// ─── Test 3: Contiguous check ───────────────────────────────────────
// Fresh tensor should be contiguous. Transposed tensor won't be (stage 2).
TEST(contiguous) {
    Tensor t({2, 3});
    assert(t.is_contiguous());
}

// ─── Test 4: Element access ─────────────────────────────────────────
// Write a value, read it back through multi-index.
// The addressing equation: offset = sum(index[i] * stride[i])
TEST(element_access) {
    Tensor t({2, 3}, Dtype::Float32);
    t.at<float>({0, 0}) = 1.0f;
    t.at<float>({0, 1}) = 2.0f;
    t.at<float>({0, 2}) = 3.0f;
    t.at<float>({1, 0}) = 4.0f;
    t.at<float>({1, 1}) = 5.0f;
    t.at<float>({1, 2}) = 6.0f;

    assert(t.at<float>({0, 0}) == 1.0f);
    assert(t.at<float>({1, 2}) == 6.0f);
}

// ─── Test 5: from_data ──────────────────────────────────────────────
// Copies raw memory into a tensor. Verify layout.
TEST(from_data) {
    float raw[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    Tensor t = Tensor::from_data(raw, {2, 3});

    assert(t.at<float>({0, 0}) == 10.0f);
    assert(t.at<float>({0, 2}) == 30.0f);
    assert(t.at<float>({1, 0}) == 40.0f);
    assert(t.at<float>({1, 2}) == 60.0f);
}

// ─── Test 6: from_vec ───────────────────────────────────────────────
TEST(from_vec) {
    auto t = Tensor::from_vec({1.0f, 2.0f, 3.0f});
    assert(t.ndim() == 1);
    assert(t.size(0) == 3);
    assert(t.at<float>({1}) == 2.0f);
}

// ─── Test 7: Negative indexing ──────────────────────────────────────
// t[-1] should work like Python.
TEST(negative_indexing) {
    float raw[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t = Tensor::from_data(raw, {2, 3});

    assert(t.at<float>({-1, -1}) == 6.0f);  // last element
    assert(t.at<float>({-2, 0}) == 1.0f);   // first element
}

// ─── Test 8: Dtype dispatch ─────────────────────────────────────────
// Int32 tensor should store ints correctly.
TEST(dtype_int32) {
    Tensor t({3}, Dtype::Int32);
    t.at<int32_t>({0}) = 42;
    t.at<int32_t>({1}) = -7;
    t.at<int32_t>({2}) = 100;

    assert(t.at<int32_t>({0}) == 42);
    assert(t.at<int32_t>({1}) == -7);
    assert(t.at<int32_t>({2}) == 100);
}

// ─── Runner ─────────────────────────────────────────────────────────

int main() {
    int passed = 0, failed = 0;

    std::cout << "╔══════════════════════════════════════╗\n";
    std::cout << "║   tensor-from-scratch · Stage 1     ║\n";
    std::cout << "╚══════════════════════════════════════╝\n\n";

    for (auto& [name, fn] : tests()) {
        try {
            fn();
            std::cout << "  ✓ " << name << "\n";
            passed++;
        } catch (const std::exception& e) {
            std::cout << "  ✗ " << name << " — " << e.what() << "\n";
            failed++;
        }
    }

    std::cout << "\n  " << passed << " passed, " << failed << " failed\n";

    if (failed == 0) {
        std::cout << "\n  🎯 Stage 1 complete! You understand:\n";
        std::cout << "     • malloc/free lifecycle\n";
        std::cout << "     • row-major stride computation\n";
        std::cout << "     • multi-index → flat offset addressing\n";
        std::cout << "     • contiguity semantics\n";
        std::cout << "     • dtype-parameterized storage\n";
        std::cout << "\n  → Ready for Stage 2: views, broadcasting, slicing\n";
    }

    return failed > 0 ? 1 : 0;
}
