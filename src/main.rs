mod autograd;
mod tensor;
mod nn;
mod plannar;

use autograd::graph::Graph;
use tensor::store::TensorStore;
use tensor::tensor::Tensor;
use nn::linear::Linear;
use nn::loss::mse;
use plannar::lifetime::compute_lifetimes;
fn main() {
    let mut store = TensorStore::new();
    let mut graph = Graph::new();

    let x_id = store.add(Tensor::new(vec![1.0, 2.0], vec![1, 2], true));
    let target_id = store.add(Tensor::new(vec![3.0, 3.0, 3.0], vec![1, 3], false));

    let linear = Linear::new(2, 3, &mut store);

    for epoch in 0..10 {
        // forward
        let out_id = linear.forward(x_id, &mut store, &mut graph);
        let loss_id = mse(out_id, target_id, &mut store, &mut graph);

        // backward
        graph.backward(&mut store, loss_id);

        // update weights
        let lr = 0.01;

        let weight = store.get_mut(linear.weight_id);
        for i in 0..weight.data.len() {
            weight.data[i] -= lr * weight.grad[i];
        }

        let bias = store.get_mut(linear.bias_id);
        for i in 0..bias.data.len() {
            bias.data[i] -= lr * bias.grad[i];
        }

        println!(
            "Epoch {} Loss: {:?}, Output: {:?}",
            epoch,
            store.get(loss_id).data,
            store.get(out_id).data
        );

        if epoch == 9 {
            let lifetimes = compute_lifetimes(&graph);

            for (id, lt) in lifetimes.iter() {
                println!("Tensor {}: birth={}, death={}", id, lt.birth, lt.death);
            }
        }

        // reset graph + grads (IMPORTANT)
        graph.nodes.clear();
        for t in &mut store.tensors {
            t.zero_grad();
        }
    }

}