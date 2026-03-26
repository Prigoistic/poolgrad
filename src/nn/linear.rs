#![allow(dead_code)]

//y=mx+c this is the linear layer defination by changing the weights and bias we can change the line and fit it to the data
use crate::tensor::store::TensorStore;
use crate::autograd::graph::Graph;
use crate::mem::pool::MemoryPool;
use crate::tensor::tensor::{matmul_scheduled_with_pool, Tensor};

pub struct Linear {
    pub weight_id: usize,
    pub bias_id: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        store: &mut TensorStore,
    ) -> Self {
        // simple initialization (no fancy init yet) we will add Xavier or Kaiming later
        let weight_data = vec![0.5; in_features * out_features];
        let bias_data = vec![0.0; out_features];

        let weight = Tensor::new(weight_data, vec![in_features, out_features], true);
        let bias = Tensor::new(bias_data, vec![1, out_features], true);

        let weight_id = store.add(weight);
        let bias_id = store.add(bias);

        Self {
            weight_id,
            bias_id,
            in_features,
            out_features,
        }
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
        // ADDING x @ w
        let matmul_out = matmul_scheduled_with_pool(input_id, self.weight_id, store, graph, pool);
        
        Tensor::add_with_pool(matmul_out, self.bias_id, store, graph, pool)
    }
}