#include "tensor.hpp"
#include <cassert>
#include <iomanip>

namespace tfs {

// ─── TODO 1: Stride computation ─────────────────────────────────────
// Given shape [2, 3, 4], produce strides [12, 4, 1] (row-major).
// Hint: walk right-to-left, accumulate products.

std::vector<int64_t> Tensor::compute_contiguous_strides(
    const std::vector<int64_t>& shape) {
    // YOUR CODE HERE
    return {};
}

// ─── TODO 2: Main constructor ───────────────────────────────────────
// Allocate storage, zero-init, set shape/strides/dtype.

Tensor::Tensor(std::vector<int64_t> shape, Dtype dtype)
    : storage_offset_(0),
      shape_(std::move(shape)),
      dtype_(dtype) {
    // YOUR CODE HERE
    // 1. Compute strides from shape_
    // 2. Calculate total bytes needed (numel * element_size)
    // 3. Create storage (shared_ptr<Storage>)
    // 4. Zero-initialize the buffer
}

// ─── PROVIDED: View constructor (you'll use this in stage 2) ────────
Tensor::Tensor(std::shared_ptr<Storage> storage, int64_t storage_offset,
               std::vector<int64_t> shape, std::vector<int64_t> strides,
               Dtype dtype)
    : storage_(std::move(storage)),
      storage_offset_(storage_offset),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      dtype_(dtype) {}

// ─── TODO 3: from_data ──────────────────────────────────────────────
// Create a tensor and copy raw bytes into it.

Tensor Tensor::from_data(const void* data, std::vector<int64_t> shape,
                         Dtype dtype) {
    // YOUR CODE HERE
    return Tensor({1});
}

// ─── TODO 4: from_vec ───────────────────────────────────────────────
Tensor Tensor::from_vec(std::initializer_list<float> vals) {
    // YOUR CODE HERE
    return Tensor({1});
}

// ─── TODO 5: numel ──────────────────────────────────────────────────
// Product of all dimensions. Empty shape = scalar = 1 element.

int64_t Tensor::numel() const {
    // YOUR CODE HERE
    return 0;
}

int64_t Tensor::size(int64_t dim) const {
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim()) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape_[dim];
}

// ─── TODO 6: is_contiguous ──────────────────────────────────────────
// Check if strides match row-major layout.
// Tricky bit: size-1 dims can have any stride (they're never stepped).

bool Tensor::is_contiguous() const {
    // YOUR CODE HERE
    return false;
}

// ─── TODO 7: flat_offset ────────────────────────────────────────────
// The core addressing equation.
// Support negative indexing (Python-style).
// Bounds-check each index.

int64_t Tensor::flat_offset(const std::vector<int64_t>& indices) const {
    // YOUR CODE HERE
    return 0;
}

// ─── TODO 8: data_ptr ───────────────────────────────────────────────
// Return pointer to first element (accounting for storage_offset).

void* Tensor::data_ptr() {
    // YOUR CODE HERE
    return nullptr;
}

const void* Tensor::data_ptr() const {
    // YOUR CODE HERE
    return nullptr;
}

// ─── PROVIDED: Debug helpers ────────────────────────────────────────

std::string Tensor::repr() const {
    std::ostringstream os;
    os << "Tensor(shape=[";
    for (int64_t i = 0; i < ndim(); ++i) {
        if (i > 0) os << ", ";
        os << shape_[i];
    }
    os << "], strides=[";
    for (int64_t i = 0; i < ndim(); ++i) {
        if (i > 0) os << ", ";
        os << strides_[i];
    }
    os << "], dtype=" << dtype_name(dtype_)
       << ", contiguous=" << (is_contiguous() ? "true" : "false")
       << ", numel=" << numel()
       << ", storage_bytes=" << storage_->nbytes()
       << ")";
    return os.str();
}

void Tensor::print(std::ostream& os) const {
    os << repr() << "\n";
}

} // namespace tfs
