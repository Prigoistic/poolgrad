use crate::autograd::node::{Node, Operation};
use crate::tensor::store::TensorStore;



#[allow(dead_code)]
pub struct Graph {
    pub nodes: Vec<Node>,

}

#[allow(dead_code)]
impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node:Node) {
        self.nodes.push(node);
    }

    pub fn backward(&self, store: &mut TensorStore, loss_id: usize) {
        // Step 1: initialize d(loss)/d(loss) = 1
        {
            let loss = store.get_mut(loss_id); 
            if loss.requires_grad {
                for g in loss.grad.iter_mut() {
                    *g = 1.0;
                }
            }
        }

        // Step 2: traverse graph in reverse topological order (construction order)
        for node in self.nodes.iter().rev() {
            match node.op {
                Operation::Add => {
                    let out_grad = store.get(node.output).grad.clone();

                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];

                    let a_req = store.get(a_id).requires_grad;
                    let b_req = store.get(b_id).requires_grad;

                    if a_id == b_id {
                        if a_req {
                            let a = store.get_mut(a_id);
                            for i in 0..out_grad.len() {
                                a.grad[i] += 2.0 * out_grad[i];
                            }
                        }
                    } else {
                        let (a, b) = store.get2_mut(a_id, b_id);
                        for i in 0..out_grad.len() {
                            if a_req {
                                a.grad[i] += out_grad[i];
                            }
                            if b_req {
                                b.grad[i] += out_grad[i];
                            }
                        }
                    }
                }

                Operation::Mul => {
                    let out_grad = store.get(node.output).grad.clone();

                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];

                    let a_data = store.get(a_id).data.clone();
                    let b_data = store.get(b_id).data.clone();

                    let a_req = store.get(a_id).requires_grad;
                    let b_req = store.get(b_id).requires_grad;

                    if a_id == b_id {
                        // y = a * a => dy/da = 2a
                        if a_req {
                            let a = store.get_mut(a_id);
                            for i in 0..out_grad.len() {
                                a.grad[i] += 2.0 * a_data[i] * out_grad[i];
                            }
                        }
                    } else {
                        let (a, b) = store.get2_mut(a_id, b_id);
                        for i in 0..out_grad.len() {
                            if a_req {
                                a.grad[i] += b_data[i] * out_grad[i];
                            }
                            if b_req {
                                b.grad[i] += a_data[i] * out_grad[i];
                            }
                        }
                    }
                }

                Operation::MatMul => {
                    let out_grad = store.get(node.output).grad.clone();

                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];

                    let a_shape = store.get(a_id).shape.clone();
                    let b_shape = store.get(b_id).shape.clone();

                    assert_eq!(a_shape.len(), 2, "A must be 2D for MatMul backward");
                    assert_eq!(b_shape.len(), 2, "B must be 2D for MatMul backward");
                    assert_eq!(a_shape[1], b_shape[0], "Inner dimensions must match");

                    let (m, n) = (a_shape[0], a_shape[1]);
                    let p = b_shape[1];

                    assert_eq!(out_grad.len(), m * p, "Output grad has wrong size");

                    let a_data = store.get(a_id).data.clone();
                    let b_data = store.get(b_id).data.clone();

                    let a_req = store.get(a_id).requires_grad;
                    let b_req = store.get(b_id).requires_grad;

                    if !a_req && !b_req {
                        continue;
                    }

                    // dA = dC @ B^T, dB = A^T @ dC
                    let mut d_a = vec![0.0f32; m * n];
                    let mut d_b = vec![0.0f32; n * p];

                    for i in 0..m {
                        for j in 0..p {
                            let go = out_grad[i * p + j];
                            for k in 0..n {
                                d_a[i * n + k] += go * b_data[k * p + j];
                                d_b[k * p + j] += a_data[i * n + k] * go;
                            }
                        }
                    }

                    if a_id == b_id {
                        // Only valid for square A (m==n==p) in practice; we still accumulate safely.
                        if a_req {
                            let a = store.get_mut(a_id);
                            for idx in 0..a.grad.len() {
                                a.grad[idx] += d_a[idx];
                                if b_req && idx < d_b.len() {
                                    a.grad[idx] += d_b[idx];
                                }
                            }
                        }
                    } else {
                        let (a, b) = store.get2_mut(a_id, b_id);
                        if a_req {
                            for idx in 0..d_a.len() {
                                a.grad[idx] += d_a[idx];
                            }
                        }
                        if b_req {
                            for idx in 0..d_b.len() {
                                b.grad[idx] += d_b[idx];
                            }
                        }
                    }
                }

                Operation::ReLU => {
                    let out_grad = store.get(node.output).grad.clone();
                    let input_id = node.inputs[0];

                    if !store.get(input_id).requires_grad {
                        continue;
                    }

                    let input_data = store.get(input_id).data.clone();
                    let input = store.get_mut(input_id);

                    for i in 0..out_grad.len() {
                        let grad = if input_data[i] > 0.0 { 1.0 } else { 0.0 };
                        input.grad[i] += grad * out_grad[i];
                    }
                }

                Operation::MSE => {
                    // loss = mean((pred - target)^2)
                    // dloss/dpred = (2/N) * (pred - target) * upstream
                    // dloss/dtarget = -(2/N) * (pred - target) * upstream
                    let out_grad = store.get(node.output).grad.clone();
                    let upstream = out_grad[0];

                    let pred_id = node.inputs[0];
                    let target_id = node.inputs[1];

                    let pred_req = store.get(pred_id).requires_grad;
                    let target_req = store.get(target_id).requires_grad;

                    let pred_data = store.get(pred_id).data.clone();
                    let target_data = store.get(target_id).data.clone();

                    assert_eq!(
                        pred_data.len(),
                        target_data.len(),
                        "MSE backward: pred and target must have same length"
                    );

                    let n = pred_data.len() as f32;
                    let scale = 2.0 / n;

                    if pred_id == target_id {
                        // (pred - pred) == 0, gradients are zero
                    } else {
                        let (pred, target) = store.get2_mut(pred_id, target_id);
                        for i in 0..pred_data.len() {
                            let diff = pred_data[i] - target_data[i];
                            let g = scale * diff * upstream;
                            if pred_req {
                                pred.grad[i] += g;
                            }
                            if target_req {
                                target.grad[i] -= g;
                            }
                        }
                    }
                }
                
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::Graph;
    use crate::tensor::store::TensorStore;
    use crate::tensor::tensor::{matmul, relu, Tensor};

    #[test]
    fn matmul_backward_matches_sums_when_loss_grad_is_ones() {
        let mut store = TensorStore::new();
        let mut graph = Graph::new();

        // A: 2x3, B: 3x2
        let a_id = store.add(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true));
        let b_id = store.add(Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], true));

        let out_id = matmul(a_id, b_id, &mut store, &mut graph);
        graph.backward(&mut store, out_id);

        // With dC = ones, dA[i,k] = sum_j B[k,j]
        let expected_a_grad = vec![15.0, 19.0, 23.0, 15.0, 19.0, 23.0];
        // With dC = ones, dB[k,j] = sum_i A[i,k]
        let expected_b_grad = vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0];

        assert_eq!(store.get(a_id).grad, expected_a_grad);
        assert_eq!(store.get(b_id).grad, expected_b_grad);
    }

    #[test]
    fn relu_backward_masks_non_positive_inputs() {
        let mut store = TensorStore::new();
        let mut graph = Graph::new();

        let x_id = store.add(Tensor::new(vec![-1.0, 2.0, 0.5, 0.0], vec![4], true));
        let y_id = relu(x_id, &mut store, &mut graph);

        graph.backward(&mut store, y_id);

        // dy/dx is 0 for x<=0, 1 for x>0 (with upstream grad = 1)
        assert_eq!(store.get(x_id).grad, vec![0.0, 1.0, 1.0, 0.0]);
    }

    
}