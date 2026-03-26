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
        }
    }

    pub fn get(&mut self, size: usize) -> Vec<f32> {
        // Track bytes checked out from the pool.
        // f32 = 4 bytes.
        let bytes = size
            .checked_mul(4)
            .expect("MemoryPool::get: size overflow when computing bytes");
        self.current_memory = self
            .current_memory
            .checked_add(bytes)
            .expect("MemoryPool::get: current_memory overflow");
        self.peak_memory = self.peak_memory.max(self.current_memory);

        let global_after = GLOBAL_CURRENT_MEMORY.fetch_add(bytes, Ordering::Relaxed) + bytes;
        update_global_peak(global_after);

        if self.enabled {
            if let Some(buffers) = self.free.get_mut(&size) {
                if let Some(mut buffer) = buffers.pop() {
                    self.reuses += 1;
                    buffer.fill(0.0);
                    return buffer;
                }
            }
        }

        self.allocations += 1;
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
            self.free.entry(size).or_insert(Vec::new()).push(buffer);
        } else {
            // In baseline mode, released buffers are dropped.
        }
    }
}