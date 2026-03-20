mod autograd;
mod tensor;

use autograd::graph::Graph;
use tensor::store::TensorStore;
use tensor::tensor::{matmul, Tensor};

fn main() {
    let mut store = TensorStore::new();
    let mut graph = Graph::new();

    // Identity matrix
    let a_id = store.add(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], true));

    // Random matrix
    let b_id = store.add(Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], true));

    let c_id = matmul(a_id, b_id, &mut store, &mut graph);

    graph.backward(&mut store, c_id);

    println!("C: {:?}", store.get(c_id).data);
    println!("Grad A: {:?}", store.get(a_id).grad);
    println!("Grad B: {:?}", store.get(b_id).grad);

}