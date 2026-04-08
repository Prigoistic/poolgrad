use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};

static TEMP_CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static TEMP_PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn update_peak(new_current: usize) {
    let mut peak = TEMP_PEAK_BYTES.load(Ordering::Relaxed);
    while new_current > peak {
        match TEMP_PEAK_BYTES.compare_exchange_weak(
            peak,
            new_current,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

#[inline]
fn temp_alloc(bytes: usize) {
    let after = TEMP_CURRENT_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    update_peak(after);
}

#[inline]
fn temp_free(bytes: usize) {
    let before = TEMP_CURRENT_BYTES.fetch_sub(bytes, Ordering::Relaxed);
    debug_assert!(before >= bytes, "TEMP_CURRENT_BYTES underflow");
}

pub fn reset() {
    TEMP_CURRENT_BYTES.store(0, Ordering::Relaxed);
    TEMP_PEAK_BYTES.store(0, Ordering::Relaxed);
}

pub fn peak_bytes() -> usize {
    TEMP_PEAK_BYTES.load(Ordering::Relaxed)
}

/// A Vec-backed temporary buffer that contributes to the global temp memory peak.
///
/// This is intentionally lightweight: it tracks capacity growth and subtracts on Drop.
#[derive(Debug)]
pub struct TrackedBufF32 {
    buf: Vec<f32>,
    tracked_bytes: usize,
}

impl TrackedBufF32 {
    pub fn zeros(len: usize) -> Self {
        let buf = vec![0.0f32; len];
        let tracked_bytes = buf
            .capacity()
            .checked_mul(std::mem::size_of::<f32>())
            .expect("TrackedBufF32 bytes overflow");
        temp_alloc(tracked_bytes);
        Self { buf, tracked_bytes }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let buf: Vec<f32> = Vec::with_capacity(capacity);
        let tracked_bytes = buf
            .capacity()
            .checked_mul(std::mem::size_of::<f32>())
            .expect("TrackedBufF32 bytes overflow");
        temp_alloc(tracked_bytes);
        Self { buf, tracked_bytes }
    }

    #[inline]
    pub fn resize(&mut self, new_len: usize, value: f32) {
        let old_cap = self.buf.capacity();
        self.buf.resize(new_len, value);
        let new_cap = self.buf.capacity();
        if new_cap > old_cap {
            let delta_elems = new_cap - old_cap;
            let delta_bytes = delta_elems
                .checked_mul(std::mem::size_of::<f32>())
                .expect("TrackedBufF32 resize bytes overflow");
            temp_alloc(delta_bytes);
            self.tracked_bytes = self
                .tracked_bytes
                .checked_add(delta_bytes)
                .expect("TrackedBufF32 tracked_bytes overflow");
        }
    }
}

impl Deref for TrackedBufF32 {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        self.buf.as_slice()
    }
}

impl DerefMut for TrackedBufF32 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buf.as_mut_slice()
    }
}

impl Drop for TrackedBufF32 {
    fn drop(&mut self) {
        temp_free(self.tracked_bytes);
    }
}
