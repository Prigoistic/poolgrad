use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static GLOBAL_CURRENT_MEMORY: AtomicUsize = AtomicUsize::new(0);
static GLOBAL_PEAK_MEMORY: AtomicUsize = AtomicUsize::new(0);

fn update_global_peak(new_current: usize) {
    let mut peak = GLOBAL_PEAK_MEMORY.load(Ordering::Relaxed);
    while new_current > peak {
        match GLOBAL_PEAK_MEMORY.compare_exchange_weak(
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

pub struct MemoryPool {
    pub enabled: bool,
    pub free: HashMap<usize, Vec<Vec<f32>>>,
    pub allocations: usize,
    pub reuses: usize,

    // Peak memory tracking (bytes checked out).
    pub current_memory: usize,
    pub peak_memory: usize,

    // Cached/free bytes kept in the pool for reuse.
    pub cached_memory: usize,
    pub cached_peak: usize,

    // Peak resident bytes = checked-out (live) + cached.
    pub resident_peak: usize,
}

impl MemoryPool {
    pub fn reset_global_metrics() {
        GLOBAL_CURRENT_MEMORY.store(0, Ordering::Relaxed);
        GLOBAL_PEAK_MEMORY.store(0, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub fn global_current_memory_bytes() -> usize {
        GLOBAL_CURRENT_MEMORY.load(Ordering::Relaxed)
    }

    pub fn global_peak_memory_bytes() -> usize {
        GLOBAL_PEAK_MEMORY.load(Ordering::Relaxed)
    }

    pub fn new() -> Self {
        Self {
            enabled: true,
            free: HashMap::new(),
            allocations: 0,
            reuses: 0,
            current_memory: 0,
            peak_memory: 0,
            cached_memory: 0,
            cached_peak: 0,
            resident_peak: 0,
        }
    }

    #[inline]
    fn update_peaks(&mut self) {
        self.peak_memory = self.peak_memory.max(self.current_memory);
        self.cached_peak = self.cached_peak.max(self.cached_memory);
        let resident = self
            .current_memory
            .checked_add(self.cached_memory)
            .expect("MemoryPool: resident bytes overflow");
        self.resident_peak = self.resident_peak.max(resident);
    }

    #[allow(dead_code)]
    pub fn resident_current_bytes(&self) -> usize {
        self.current_memory + self.cached_memory
    }

    pub fn get(&mut self, size: usize) -> Vec<f32> {
        // Track bytes checked out from the pool.
        // f32 = 4 bytes.
        let bytes = size
            .checked_mul(4)
            .expect("MemoryPool::get: size overflow when computing bytes");
        if self.enabled {
            if let Some(buffers) = self.free.get_mut(&size) {
                if let Some(mut buffer) = buffers.pop() {
                    self.reuses += 1;

                    self.cached_memory = self
                        .cached_memory
                        .checked_sub(bytes)
                        .expect("MemoryPool::get: cached_memory underflow");

                    self.current_memory = self
                        .current_memory
                        .checked_add(bytes)
                        .expect("MemoryPool::get: current_memory overflow");

                    let global_after = GLOBAL_CURRENT_MEMORY.fetch_add(bytes, Ordering::Relaxed) + bytes;
                    update_global_peak(global_after);

                    self.update_peaks();

                    buffer.fill(0.0);
                    return buffer;
                }
            }
        }

        // Fresh allocation.
        self.allocations += 1;
        self.current_memory = self
            .current_memory
            .checked_add(bytes)
            .expect("MemoryPool::get: current_memory overflow");

        let global_after = GLOBAL_CURRENT_MEMORY.fetch_add(bytes, Ordering::Relaxed) + bytes;
        update_global_peak(global_after);
        self.update_peaks();

        vec![0.0; size]
    }

    #[allow(dead_code)]
    pub fn get_no_clear(&mut self, size: usize) -> Vec<f32> {
        // Identical accounting to `get`, but does not clear reused buffers.
        // Only use when the caller overwrites every element before reading.
        let bytes = size
            .checked_mul(4)
            .expect("MemoryPool::get_no_clear: size overflow when computing bytes");

        if self.enabled {
            if let Some(buffers) = self.free.get_mut(&size) {
                if let Some(buffer) = buffers.pop() {
                    self.reuses += 1;

                    self.cached_memory = self
                        .cached_memory
                        .checked_sub(bytes)
                        .expect("MemoryPool::get_no_clear: cached_memory underflow");

                    self.current_memory = self
                        .current_memory
                        .checked_add(bytes)
                        .expect("MemoryPool::get_no_clear: current_memory overflow");

                    let global_after = GLOBAL_CURRENT_MEMORY.fetch_add(bytes, Ordering::Relaxed) + bytes;
                    update_global_peak(global_after);
                    self.update_peaks();

                    return buffer;
                }
            }
        }

        // Fresh allocation.
        self.allocations += 1;
        self.current_memory = self
            .current_memory
            .checked_add(bytes)
            .expect("MemoryPool::get_no_clear: current_memory overflow");

        let global_after = GLOBAL_CURRENT_MEMORY.fetch_add(bytes, Ordering::Relaxed) + bytes;
        update_global_peak(global_after);
        self.update_peaks();

        vec![0.0; size]
    }

    pub fn release(&mut self, buffer: Vec<f32>) {
        let size = buffer.len();
        let bytes = size
            .checked_mul(4)
            .expect("MemoryPool::release: size overflow when computing bytes");
        self.current_memory = self
            .current_memory
            .checked_sub(bytes)
            .expect("MemoryPool::release: current_memory underflow (released more than checked out)");

        let global_before = GLOBAL_CURRENT_MEMORY.fetch_sub(bytes, Ordering::Relaxed);
        debug_assert!(
            global_before >= bytes,
            "GLOBAL_CURRENT_MEMORY underflow (released more than checked out globally)"
        );

        if self.enabled {
            self.cached_memory = self
                .cached_memory
                .checked_add(bytes)
                .expect("MemoryPool::release: cached_memory overflow");
            self.free.entry(size).or_insert(Vec::new()).push(buffer);
            self.update_peaks();
        } else {
            // In baseline mode, released buffers are dropped.
            self.update_peaks();
        }
    }
}