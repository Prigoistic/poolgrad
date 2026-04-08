use crate::autograd::graph::Graph;
use crate::autograd::node::{Node, Operation};
use crate::memory::pool::MemoryPool;
use crate::tensor::store::TensorStore;
use crate::tensor::tensor::Tensor;

pub fn mse(
    pred_id: usize,
    target_id: usize,
    store: &mut TensorStore,
    graph: &mut Graph,
    _pool: &mut MemoryPool,
) -> usize {
    let pred = store.get(pred_id);
    let target = store.get(target_id);

    assert_eq!(
        pred.data.len(),
        target.data.len(),
        "mse: pred and target must have the same number of elements"
    );

    let len = pred.data.len();
    let mean = pred
        .data
        .iter()
        .zip(&target.data)
        .map(|(p, t)| {
            let d = p - t;
            d * d
        })
        .sum::<f32>()
        / len as f32;

    let loss = Tensor::new(vec![mean], vec![1], true);

    let loss_id = store.add(loss);

    let node = Node {
        input0: pred_id,
        input1: Some(target_id),
        output: loss_id,
        op: Operation::MSE,
    };

    let node_id = graph.add_node(node);
    store.get_mut(loss_id).creator = Some(node_id);

    loss_id
}
