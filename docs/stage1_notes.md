# Stage 1 — Interview talking points

## Questions you can now answer cold

**Q: "What are strides in a tensor?"**
Strides tell you how many elements to skip in the underlying 1D buffer to move one step along each dimension. For a shape `[2, 3, 4]` tensor in row-major layout, strides are `[12, 4, 1]` — moving along dim-0 skips `3*4=12` elements, dim-1 skips `4`, dim-2 is contiguous.

**Q: "What does .contiguous() do in PyTorch?"**
It checks if the tensor's strides match what row-major layout would produce. If not (e.g. after a `.transpose()`), it allocates new storage and copies data into contiguous order. The check itself is O(ndim) — just walk the strides right-to-left verifying each equals the product of sizes to its right.

**Q: "Why does PyTorch separate Storage from Tensor?"**
So that multiple Tensors can share the same underlying buffer. A `.view()` creates a new Tensor with different shape/strides but pointing at the same Storage. This makes reshape, transpose, and slicing essentially free (no data copy). The cost: you need to be careful about in-place ops aliasing.

**Q: "What's the addressing equation for element [i, j, k]?"**
`byte_offset = (storage_offset + i*stride[0] + j*stride[1] + k*stride[2]) * element_size`

This is the single equation that all of numpy/PyTorch's view machinery reduces to.

## What I built

- `Storage` class: raw `malloc`/`free`, byte buffer, non-copyable (move-only)
- `Tensor` class: shape, strides, dtype, shared_ptr to Storage
- Contiguity check: walk strides right-to-left
- Multi-index → flat offset with negative indexing support
- Dtype dispatch: float32, float64, int32, int64

## What I'd improve next

- [ ] Pretty-print with proper nesting (numpy-style)
- [ ] Bounds checking on storage access
- [ ] Scalar tensor support (0-dim)
