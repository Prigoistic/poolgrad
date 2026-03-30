use crate::autograd::graph::Graph;
use std::collections::HashMap;

/// Lifetime (birth/death) for a tensor id within a single `Graph`.
///
/// - `birth`: earliest node step where the tensor appears as an output.
/// - `death`: last node step where the tensor is used as an input.
///
/// Used to release intermediate gradient buffers back into the `MemoryPool` once
/// they are no longer needed.
#[derive(Debug, Clone, Copy)]
pub struct Lifetime {
    pub birth: usize,
    pub death: usize,
}

pub fn compute_lifetimes(graph: &Graph) -> HashMap<usize, Lifetime> {
    let mut lifetimes: HashMap<usize, Lifetime> = HashMap::new();

    for (step, node) in graph.nodes.iter().enumerate() {
        // Output tensor: mark/refresh birth.
        lifetimes
            .entry(node.output)
            .and_modify(|l| l.birth = l.birth.min(step))
            .or_insert(Lifetime {
                birth: step,
                death: step,
            });

        // Input tensors: extend death.
        lifetimes
            .entry(node.input0)
            .and_modify(|l| l.death = l.death.max(step))
            .or_insert(Lifetime {
                birth: 0,
                death: step,
            });

        if let Some(input1) = node.input1 {
            lifetimes
                .entry(input1)
                .and_modify(|l| l.death = l.death.max(step))
                .or_insert(Lifetime {
                    birth: 0,
                    death: step,
                });
        }
    }

    lifetimes
}
