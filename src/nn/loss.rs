#![allow(dead_code)]

use crate::autograd::graph::Graph;
use crate::autograd::node::{Node, Operation};
use crate::mem::pool::MemoryPool;
use crate::tensor::store::TensorStore;
use crate::tensor::tensor::Tensor;

pub fn mse(
    pred_id: usize,
    target_id: usize,
    store: &mut TensorStore,
    graph: &mut Graph,
    pool: &mut MemoryPool,
) -> usize {
    let pred = store.get(pred_id);
    let target = store.get(target_id);

    assert_eq!(
        pred.data.len(),
        target.data.len(),
        "mse: pred and target must have the same number of elements"
    );

    let diff_data: Vec<f32> = pred
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .collect();

    let mean = diff_data.iter().sum::<f32>() / diff_data.len() as f32;

    let loss = Tensor::new_with_pool(vec![mean], vec![1], true, pool);

    let loss_id = store.add(loss);

    let node = Node {
        input0: pred_id,
        input1: Some(target_id),
        output: loss_id,
        op: Operation::MSE,
    };

    let node_id = graph.nodes.len();
    graph.add_node(node);
    store.get_mut(loss_id).creator = Some(node_id);

    loss_id
}
