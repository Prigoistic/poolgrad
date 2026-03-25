#![allow(dead_code)]

use std::collections::HashMap;
use crate::autograd::graph::Graph;

#[derive(Debug, Clone)]
pub struct Lifetime {
    pub birth: usize,
    pub death: usize,
}

pub fn compute_lifetimes(graph: &Graph) -> HashMap<usize, Lifetime> {
    let mut lifetimes: HashMap<usize, Lifetime> = HashMap::new();

    for (step, node) in graph.nodes.iter().enumerate() {
        // OUTPUT tensor → birth
        lifetimes
            .entry(node.output)
            .and_modify(|l| l.birth = l.birth.min(step))
            .or_insert(Lifetime {
                birth: step,
                death: step,
            });

        // INPUT tensors → update death
        for &input in &node.inputs {
            lifetimes
                .entry(input)
                .and_modify(|l| l.death = l.death.max(step))
                .or_insert(Lifetime {
                    birth: 0,
                    death: step,
                });
        }
    }

    lifetimes
}
