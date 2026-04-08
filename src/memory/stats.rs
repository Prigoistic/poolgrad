use crate::tensor::store::TensorStore;

/// A point-in-time memory breakdown used for reporting.
///
/// Definitions:
/// - `grad_reserved_bytes`: peak bytes reserved for gradient buffers (e.g. pool resident bytes or
///   planner pre-allocated buffers). This is the *footprint* you pay for reuse.
/// - `grad_live_bytes`: peak bytes actively checked out / in-use during backward.
/// - `activation_bytes`: bytes held by forward activations in the `TensorStore` (Vec capacity).
/// - `temp_bytes`: peak bytes tracked by `TrackedBufF32` temporaries.
#[derive(Debug, Default, Clone, Copy)]
pub struct MemoryBytes {
    pub grad_reserved_bytes: usize,
    pub grad_live_bytes: usize,
    pub activation_bytes: usize,
    pub temp_bytes: usize,
}

impl MemoryBytes {
    /// Total reserved footprint: activation + grad_reserved + temp.
    #[inline]
    pub fn total_reserved_bytes(&self) -> usize {
        self.activation_bytes
            .checked_add(self.grad_reserved_bytes)
            .and_then(|x| x.checked_add(self.temp_bytes))
            .expect("MemoryBytes: total bytes overflow")
    }
}

#[inline]
pub fn activation_bytes(store: &TensorStore) -> usize {
    store
        .tensors
        .iter()
        .map(|t| {
            t.data
                .capacity()
                .checked_mul(std::mem::size_of::<f32>())
                .expect("activation bytes overflow")
        })
        .sum()
}
