/// A fully-connected linear layer: `y = x @ W + b`.
use crate::autograd::graph::Graph;
use crate::memory::pool::MemoryPool;
use crate::tensor::store::TensorStore;
use crate::tensor::tensor::{Tensor, add_with_pool, matmul_with_pool};

pub struct Linear {
    pub weight_id: usize,
    pub bias_id: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, store: &mut TensorStore) -> Self {
        // Simple constant init. (Initialization strategy is intentionally out of scope here.)
        let weight_data = vec![0.5; in_features * out_features];
        let bias_data = vec![0.0; out_features];

        let weight = Tensor::new(weight_data, vec![in_features, out_features], true);
        let bias = Tensor::new(bias_data, vec![1, out_features], true);

        let weight_id = store.add(weight);
        let bias_id = store.add(bias);

        Self { weight_id, bias_id }
    }
}

impl Linear {
    pub fn forward(
        &self,
        input_id: usize,
        store: &mut TensorStore,
        graph: &mut Graph,
        pool: &mut MemoryPool,
    ) -> usize {
        let matmul_out = matmul_with_pool(input_id, self.weight_id, store, graph, pool);
        add_with_pool(matmul_out, self.bias_id, store, graph, pool)
    }
}
