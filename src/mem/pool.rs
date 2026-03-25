use std::collections::HashMap;

pub struct MemoryPool {
    pub enabled: bool,
    pub free: HashMap<usize, Vec<Vec<f32>>>,
    pub allocations: usize,
    pub reuses: usize,

    // Peak memory tracking (in number of f32 elements).
    pub current_memory: usize,
    pub peak_memory: usize,
}

impl MemoryPool {
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
        self.current_memory += size;
        self.peak_memory = self.peak_memory.max(self.current_memory);
        vec![0.0; size]
    }

    pub fn release(&mut self, buffer: Vec<f32>) {
        let size = buffer.len();
        if self.enabled {
            self.free.entry(size).or_insert(Vec::new()).push(buffer);
        } else {
            // In baseline mode, released buffers are dropped.
            self.current_memory = self.current_memory.saturating_sub(size);
        }
    }
}