mod autograd;
mod tensor;

use autograd::graph::Graph;
use tensor::tensor::{Tensor, TensorStore};

fn main() {
    let mut store = TensorStore::new();
    let mut graph = Graph::new();

    let a_id = store.add(Tensor::new(vec![1.0, 2.0], vec![2], true));
    let b_id = store.add(Tensor::new(vec![3.0, 4.0], vec![2], true));

    let c_id = Tensor::add(a_id, b_id, &mut store, &mut graph);

    println!("Result: {}", store.get(c_id));
    println!("Graph nodes: {}", graph.nodes.len());
    
}