//! Backward liveness information used by the gradient buffer planner.

/// Backward lifetime for a tensor's *gradient buffer* within a single backward pass.
///
/// Semantics are defined over the reverse traversal of the active subgraph:
///
/// - `birth`: earliest backward step when the grad buffer must exist (first write/seed).
/// - `last_use`: latest backward step when the grad buffer is still needed.
///   For non-leaf intermediates, this is typically the step where its creator node consumes
///   the upstream gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifetime {
    pub birth: usize,
    pub last_use: usize,
}

impl Lifetime {
    #[inline]
    pub fn new(step: usize) -> Self {
        Self {
            birth: step,
            last_use: step,
        }
    }

    #[inline]
    pub fn update_birth(&mut self, step: usize) {
        self.birth = self.birth.min(step);
    }

    #[inline]
    pub fn update_last_use(&mut self, step: usize) {
        self.last_use = self.last_use.max(step);
    }
}
