#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace tfs { // tensor-from-scratch

// ─── Dtype ───────────────────────────────────────────────────────────
// Interview signal: why frameworks need dtype dispatch, how storage
// is decoupled from interpretation.

enum class Dtype {
    Float32,
    Float64,
    Int32,
    Int64,
    // Stage 4+: Float16, BFloat16
};

inline size_t dtype_size(Dtype dt) {
    switch (dt) {
        case Dtype::Float32: return 4;
        case Dtype::Float64: return 8;
        case Dtype::Int32:   return 4;
        case Dtype::Int64:   return 8;
    }
    return 0;
}

inline const char* dtype_name(Dtype dt) {
    switch (dt) {
        case Dtype::Float32: return "float32";
        case Dtype::Float64: return "float64";
        case Dtype::Int32:   return "int32";
        case Dtype::Int64:   return "int64";
    }
    return "unknown";
}

// ─── Storage ─────────────────────────────────────────────────────────
// The raw memory buffer. Shared across views (refcounted via shared_ptr).
// Interview signal: PyTorch's Storage vs Tensor split, why views are
// cheap, why .contiguous() sometimes copies.

class Storage {
public:
    Storage(size_t nbytes)
        : data_(static_cast<uint8_t*>(std::malloc(nbytes))),
          nbytes_(nbytes) {
        if (!data_ && nbytes > 0) {
            throw std::bad_alloc();
        }
    }

    ~Storage() {
        std::free(data_);
    }

    // Non-copyable, movable
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&& other) noexcept
        : data_(other.data_), nbytes_(other.nbytes_) {
        other.data_ = nullptr;
        other.nbytes_ = 0;
    }

    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    size_t nbytes() const { return nbytes_; }

private:
    uint8_t* data_;
    size_t nbytes_;
};

// ─── Tensor ──────────────────────────────────────────────────────────
// Stage 1: shape, strides, dtype, contiguous check, element access.
//
// Key design decisions you should be able to defend in an interview:
//   1. Storage is shared_ptr — enables zero-copy views (stage 2)
//   2. Strides are in *elements*, not bytes — matches PyTorch convention
//   3. storage_offset_ lets slices point into the middle of a buffer

class Tensor {
public:
    // ── Construction ──

    // Empty tensor with given shape, row-major (C-contiguous) strides
    Tensor(std::vector<int64_t> shape, Dtype dtype = Dtype::Float32);

    // From raw data (copies into owned storage)
    static Tensor from_data(const void* data, std::vector<int64_t> shape,
                            Dtype dtype = Dtype::Float32);

    // Convenience: 1D tensor from initializer list
    static Tensor from_vec(std::initializer_list<float> vals);

    // ── Shape / layout ──

    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t size(int64_t dim) const;
    int64_t numel() const;
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    Dtype dtype() const { return dtype_; }

    // A tensor is contiguous if its strides match row-major layout
    // (rightmost dimension has stride 1, each dimension left is
    //  product of all dimensions to its right).
    bool is_contiguous() const;

    // ── Element access ──

    // Multi-index access: t.at({i, j, k})
    // Returns pointer to element — caller casts to the right type.
    void* data_ptr();
    const void* data_ptr() const;

    // Flat offset from multi-index (in elements, not bytes)
    int64_t flat_offset(const std::vector<int64_t>& indices) const;

    // Typed accessors — convenience for stage 1 testing
    template <typename T>
    T& at(const std::vector<int64_t>& indices) {
        return *reinterpret_cast<T*>(
            static_cast<uint8_t*>(data_ptr()) +
            flat_offset(indices) * static_cast<int64_t>(dtype_size(dtype_))
        );
    }

    template <typename T>
    const T& at(const std::vector<int64_t>& indices) const {
        return *reinterpret_cast<const T*>(
            static_cast<const uint8_t*>(data_ptr()) +
            flat_offset(indices) * static_cast<int64_t>(dtype_size(dtype_))
        );
    }

    // ── Debug ──

    // e.g. "Tensor(shape=[2, 3], strides=[3, 1], dtype=float32)"
    std::string repr() const;

    // Print elements (flat for now, pretty-print later)
    void print(std::ostream& os = std::cout) const;

    // ── Internals (public for stage 2 view construction) ──

    std::shared_ptr<Storage> storage() const { return storage_; }
    int64_t storage_offset() const { return storage_offset_; }

private:
    // Private constructor for view construction (stage 2)
    Tensor(std::shared_ptr<Storage> storage, int64_t storage_offset,
           std::vector<int64_t> shape, std::vector<int64_t> strides,
           Dtype dtype);

    std::shared_ptr<Storage> storage_;
    int64_t storage_offset_ = 0;  // offset in elements
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    Dtype dtype_;

    // Compute row-major strides from shape
    static std::vector<int64_t> compute_contiguous_strides(
        const std::vector<int64_t>& shape);
};

} // namespace tfs
