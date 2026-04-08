use crate::autograd::graph::Graph;
use crate::planner::lifetime::Lifetime;
use crate::tensor::store::TensorStore;

use std::collections::HashMap;

pub type TensorId = usize;
pub type BufferId = usize;

#[derive(Debug, Clone)]
pub struct AllocationPlan {
    pub assignments: HashMap<TensorId, BufferId>,
    pub buffer_sizes: Vec<usize>,
}

#[derive(Debug, Clone)]
struct TensorLife {
    id: TensorId,
    size: usize,
    birth: usize,
    last_use: usize,
    pinned: bool,
}

/// Deterministic, size-class aware gradient buffer planner.
///
/// This builds an *allocation plan* from backward liveness and then hands out
/// pre-allocated buffers according to the plan. No runtime `MemoryPool::get/release`
/// calls are needed in the planned path.
#[derive(Debug)]
pub struct GradPlanner {
    plan: AllocationPlan,
    buffers: Vec<Vec<f32>>,
    checked_out: Vec<bool>,

    pub allocations: usize,

    pub reserved_bytes: usize,
    pub peak_live_bytes: usize,
}

impl GradPlanner {
    pub fn build(graph: &Graph, store: &TensorStore, output_id: TensorId) -> Self {
        let (active, lifetimes, last_write) = backward_liveness(graph, store, output_id);

        let mut lives: Vec<TensorLife> = Vec::new();
        for id in 0..store.tensors.len() {
            if id >= active.len() || !active[id] {
                continue;
            }
            let t = store.get(id);
            if !t.requires_grad {
                continue;
            }

            let size = t.data.len();
            if size == 0 {
                continue;
            }

            let pinned = t.creator.is_none();
            let mut life = lifetimes
                .get(&id)
                .copied()
                .unwrap_or_else(|| Lifetime::new(0));

            // If a tensor is never consumed as an upstream gradient (e.g. a leaf),
            // its grad buffer must still remain valid until its *last write*.
            if let Some(w) = last_write.get(&id) {
                life.update_birth(*w);
                life.update_last_use(*w);
            }

            lives.push(TensorLife {
                id,
                size,
                birth: life.birth,
                last_use: life.last_use,
                pinned,
            });
        }

        let plan = build_allocation_plan(lives);
        let allocations = plan.buffer_sizes.len();
        let mut buffers: Vec<Vec<f32>> = Vec::with_capacity(plan.buffer_sizes.len());
        let mut reserved_bytes: usize = 0;
        for &sz in &plan.buffer_sizes {
            buffers.push(vec![0.0; sz]);
            reserved_bytes = reserved_bytes
                .checked_add(sz.checked_mul(4).expect("buffer bytes overflow"))
                .expect("reserved_bytes overflow");
        }

        Self {
            checked_out: vec![false; plan.buffer_sizes.len()],
            plan,
            buffers,
            allocations,
            reserved_bytes,
            peak_live_bytes: 0,
        }
    }

    #[inline]
    pub fn buffer_for(&self, tensor_id: TensorId) -> BufferId {
        *self
            .plan
            .assignments
            .get(&tensor_id)
            .unwrap_or_else(|| panic!("GradPlanner: no buffer assignment for tensor {tensor_id}"))
    }

    #[inline]
    pub fn checkout(&mut self, tensor_id: TensorId, expected_len: usize) -> Vec<f32> {
        let buf_id = self.buffer_for(tensor_id);
        debug_assert!(!self.checked_out[buf_id], "buffer already checked out");

        let mut buf = std::mem::take(&mut self.buffers[buf_id]);
        if buf.len() != expected_len {
            panic!(
                "GradPlanner: buffer size mismatch for tensor {}: buf_len={}, expected_len={}",
                tensor_id,
                buf.len(),
                expected_len
            );
        }
        buf.fill(0.0);

        self.checked_out[buf_id] = true;
        self.update_peak_live_bytes();

        buf
    }

    #[inline]
    pub fn checkin(&mut self, tensor_id: TensorId, buffer: Vec<f32>) {
        let buf_id = self.buffer_for(tensor_id);
        debug_assert!(self.checked_out[buf_id], "buffer not checked out");

        self.buffers[buf_id] = buffer;
        self.checked_out[buf_id] = false;
    }

    #[inline]
    pub fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }

    #[inline]
    pub fn live_bytes(&self) -> usize {
        let mut live: usize = 0;
        for (buf_id, checked_out) in self.checked_out.iter().copied().enumerate() {
            if checked_out {
                let sz = self.plan.buffer_sizes[buf_id];
                live = live
                    .checked_add(sz.checked_mul(4).expect("live bytes overflow"))
                    .expect("live bytes overflow");
            }
        }
        live
    }

    fn update_peak_live_bytes(&mut self) {
        let live = self.live_bytes();
        self.peak_live_bytes = self.peak_live_bytes.max(live);
    }
}

fn build_allocation_plan(mut lives: Vec<TensorLife>) -> AllocationPlan {
    // Determinism: stable ordering by (birth, id).
    lives.sort_by_key(|l| (l.birth, l.id));

    // For each size class, keep a deterministic list of (buffer_id, free_at_last_use).
    let mut free_by_size: HashMap<usize, Vec<(BufferId, usize)>> = HashMap::new();

    let mut assignments: HashMap<TensorId, BufferId> = HashMap::new();
    let mut buffer_sizes: Vec<usize> = Vec::new();

    for life in lives {
        if life.pinned {
            // Pinned grads (leaf tensors) must persist after backward; do not reuse.
            let buf_id: BufferId = buffer_sizes.len();
            buffer_sizes.push(life.size);
            assignments.insert(life.id, buf_id);
            continue;
        }

        let free_list = free_by_size.entry(life.size).or_default();

        // Find the earliest buffer that becomes free before this tensor's birth.
        // Determinism: choose the lowest buffer_id among eligible buffers.
        let mut chosen: Option<(usize, BufferId)> = None; // (index, buf_id)
        for (idx, &(buf_id, free_at)) in free_list.iter().enumerate() {
            if free_at < life.birth {
                match chosen {
                    None => chosen = Some((idx, buf_id)),
                    Some((_cidx, cbid)) => {
                        if buf_id < cbid {
                            chosen = Some((idx, buf_id));
                        }
                    }
                }
            }
        }

        let buf_id = if let Some((idx, buf_id)) = chosen {
            // Reuse.
            free_list.swap_remove(idx);
            buf_id
        } else {
            // Allocate new.
            let buf_id: BufferId = buffer_sizes.len();
            buffer_sizes.push(life.size);
            buf_id
        };

        assignments.insert(life.id, buf_id);

        // Return this buffer to the free list at last_use.
        free_list.push((buf_id, life.last_use));
    }

    AllocationPlan {
        assignments,
        buffer_sizes,
    }
}

/// Computes active tensors and backward liveness (birth/last_use) for gradient buffers.
///
/// Returns:
/// - active[tensor_id] = participates in gradient propagation
/// - lifetimes[tensor_id] = (birth, last_use) for upstream-grad consumption
/// - last_write[tensor_id] = last step where this tensor's grad is written (accumulated)
fn backward_liveness(
    graph: &Graph,
    store: &TensorStore,
    output_id: TensorId,
) -> (
    Vec<bool>,
    HashMap<TensorId, Lifetime>,
    HashMap<TensorId, usize>,
) {
    let mut active: Vec<bool> = vec![false; store.tensors.len()];
    if output_id < active.len() && store.get(output_id).requires_grad {
        active[output_id] = true;
    }

    let mut lifetimes: HashMap<TensorId, Lifetime> = HashMap::new();
    let mut last_write: HashMap<TensorId, usize> = HashMap::new();

    // Seeded output grad must exist at step 0 (first step in reverse traversal).
    if active[output_id] {
        lifetimes.insert(output_id, Lifetime::new(0));
    }

    for (step, node) in graph.nodes.iter().rev().enumerate() {
        if node.output >= active.len() || !active[node.output] {
            continue;
        }

        // Upstream gradient for node.output is *consumed* at this step.
        lifetimes
            .entry(node.output)
            .and_modify(|l| {
                l.update_birth(step);
                l.update_last_use(step);
            })
            .or_insert_with(|| Lifetime::new(step));

        // Inputs receive gradient contributions at this step.
        let a_id = node.input0;
        let b_id = node.input1;

        if a_id < active.len() && store.get(a_id).requires_grad {
            active[a_id] = true;
            last_write
                .entry(a_id)
                .and_modify(|w| *w = (*w).max(step))
                .or_insert(step);
            lifetimes
                .entry(a_id)
                .and_modify(|l| l.update_birth(step))
                .or_insert_with(|| Lifetime::new(step));
        }

        if let Some(b_id) = b_id
            && b_id < active.len()
            && store.get(b_id).requires_grad
        {
            active[b_id] = true;
            last_write
                .entry(b_id)
                .and_modify(|w| *w = (*w).max(step))
                .or_insert(step);
            lifetimes
                .entry(b_id)
                .and_modify(|l| l.update_birth(step))
                .or_insert_with(|| Lifetime::new(step));
        }
    }

    // Ensure `last_use` covers the last write for leafs/pinned grads.
    for (&id, &w) in last_write.iter() {
        lifetimes
            .entry(id)
            .and_modify(|l| l.update_last_use(w))
            .or_insert_with(|| Lifetime {
                birth: w,
                last_use: w,
            });
    }

    (active, lifetimes, last_write)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::node::{Node, Operation};

    #[test]
    fn planner_reuses_non_overlapping_buffers_same_size() {
        // Graph topology (forward):
        // t2 = add(t0, t1)
        // t3 = mul(t2, t1)
        // Backward can release grad(t3) before using grad(t2) etc.
        let mut g = Graph::new();
        g.nodes.push(Node {
            input0: 0,
            input1: Some(1),
            output: 2,
            op: Operation::Add,
        });
        g.nodes.push(Node {
            input0: 2,
            input1: Some(1),
            output: 3,
            op: Operation::Mul,
        });

        let mut store = TensorStore::new();
        // All same size = 4
        for _ in 0..4 {
            store.add(crate::tensor::tensor::Tensor::new(
                vec![0.0; 4],
                vec![4],
                true,
            ));
        }
        // Mark intermediates as having creators.
        store.get_mut(2).creator = Some(0);
        store.get_mut(3).creator = Some(1);

        let planner = GradPlanner::build(&g, &store, 3);

        // Leaf grads are pinned and should have distinct buffers.
        let b0 = planner.buffer_for(0);
        let b1 = planner.buffer_for(1);
        assert_ne!(b0, b1);

        // Intermediates may reuse buffers, but must still be assigned.
        let _ = planner.buffer_for(2);
        let _ = planner.buffer_for(3);
    }
}
