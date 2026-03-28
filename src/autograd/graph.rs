use crate::autograd::node::{Node, Operation};
use crate::mem::pool::MemoryPool;
use crate::tensor::store::TensorStore;

use crate::kernels::naive::{
    matmul_naive_add_into_slices_a_transposed, matmul_naive_add_into_slices_b_transposed,
};
use crate::kernels::selector::{KernelType, select_kernel};
use crate::kernels::tiled::{
    matmul_tiled_add_into_slices_a_transposed, matmul_tiled_add_into_slices_b_transposed,
};
use crate::kernels::tiled_mp::{
    matmul_tiled_mp_add_into_slices_a_transposed, matmul_tiled_mp_add_into_slices_b_transposed,
};

use std::collections::HashMap;

#[allow(dead_code)]
pub struct Graph {
    pub nodes: Vec<Node>,
}

#[allow(dead_code)]
impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    pub fn backward(&self, store: &mut TensorStore, loss_id: usize, _pool: &mut MemoryPool) {
        // Contract: `backward(loss_id)` seeds scalar loss with 1.0.
        // For non-scalar outputs, use `backward_seeded`.
        {
            let loss = store.get_mut(loss_id);
            if loss.requires_grad {
                assert_eq!(
                    loss.grad.len(),
                    1,
                    "Graph::backward(loss_id): loss must be scalar; use backward_seeded for non-scalar"
                );
                loss.grad[0] = 1.0;
            }
        }

        self.backward_internal(store, loss_id);
    }

    /// Clears gradients for tensors referenced by this graph, then runs `backward`.
    pub fn backward_zeroed(&self, store: &mut TensorStore, loss_id: usize, pool: &mut MemoryPool) {
        self.zero_grads_for_graph(store, pool);
        self.backward(store, loss_id, pool);
    }

    pub fn backward_seeded(
        &self,
        store: &mut TensorStore,
        output_id: usize,
        seed: &[f32],
        _pool: &mut MemoryPool,
    ) {
        {
            let out = store.get_mut(output_id);
            if out.requires_grad {
                assert_eq!(
                    out.grad.len(),
                    seed.len(),
                    "Graph::backward_seeded: seed length must match output grad length"
                );
                out.grad.copy_from_slice(seed);
            }
        }

        self.backward_internal(store, output_id);
    }

    /// Clears gradients for tensors referenced by this graph, then runs `backward_seeded`.
    pub fn backward_seeded_zeroed(
        &self,
        store: &mut TensorStore,
        output_id: usize,
        seed: &[f32],
        pool: &mut MemoryPool,
    ) {
        self.zero_grads_for_graph(store, pool);
        self.backward_seeded(store, output_id, seed, pool);
    }

    fn zero_grads_for_graph(&self, store: &mut TensorStore, pool: &mut MemoryPool) {
        // Only touch tensors referenced by this graph (inputs + outputs).
        let mut touched = vec![false; store.tensors.len()];
        for node in &self.nodes {
            if node.output < touched.len() {
                touched[node.output] = true;
            }

            if node.input0 < touched.len() {
                touched[node.input0] = true;
            }
            if let Some(id) = node.input1
                && id < touched.len()
            {
                touched[id] = true;
            }
        }

        for (id, is_touched) in touched.into_iter().enumerate() {
            if !is_touched {
                continue;
            }

            let t = store.get_mut(id);
            if !t.requires_grad {
                continue;
            }

            // If a grad buffer was released to the pool (planner), re-acquire it.
            if t.grad.is_empty() {
                t.grad = pool.get(t.data.len());
            } else {
                debug_assert_eq!(
                    t.grad.len(),
                    t.data.len(),
                    "Tensor {}: grad len != data len",
                    id
                );
                t.grad.fill(0.0);
            }
        }
    }

    fn backward_internal(&self, store: &mut TensorStore, output_id: usize) {
        // Track which tensors have an upstream gradient path from `output_id`.
        let mut active: Vec<bool> = vec![false; store.tensors.len()];
        if output_id < active.len() && store.get(output_id).requires_grad {
            active[output_id] = true;
        }

        // Cache kernel selection for repeated MatMul shapes within this backward pass.
        let mut matmul_kernel_cache: HashMap<(usize, usize, usize), KernelType> = HashMap::new();

        for node in self.nodes.iter().rev() {
            if node.output >= active.len() || !active[node.output] {
                continue;
            }

            match node.op {
                Operation::Add => {
                    let a_id = node.input0;
                    let b_id = node.input1.expect("Add op must have 2 inputs");

                    let a_req = store.get(a_id).requires_grad;
                    let b_req = store.get(b_id).requires_grad;
                    if !a_req && !b_req {
                        continue;
                    }

                    if a_req && a_id < active.len() {
                        active[a_id] = true;
                    }
                    if b_req && b_id < active.len() {
                        active[b_id] = true;
                    }

                    if a_id == b_id {
                        if a_req {
                            let (a, out) = store.get_mut_and_1(a_id, node.output);
                            assert_eq!(
                                a.grad.len(),
                                a.data.len(),
                                "Add backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                                a_id
                            );
                            debug_assert_eq!(
                                out.grad.len(),
                                a.grad.len(),
                                "Add backward: grad size mismatch"
                            );
                            for i in 0..out.grad.len() {
                                a.grad[i] += 2.0 * out.grad[i];
                            }
                        }
                    } else {
                        let (a, b, out) = store.get2_mut_and_1(a_id, b_id, node.output);
                        if a_req {
                            assert_eq!(
                                a.grad.len(),
                                a.data.len(),
                                "Add backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                                a_id
                            );
                        }
                        if b_req {
                            assert_eq!(
                                b.grad.len(),
                                b.data.len(),
                                "Add backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                                b_id
                            );
                        }
                        debug_assert_eq!(
                            out.grad.len(),
                            a.grad.len(),
                            "Add backward: grad size mismatch"
                        );
                        debug_assert_eq!(
                            out.grad.len(),
                            b.grad.len(),
                            "Add backward: grad size mismatch"
                        );
                        for i in 0..out.grad.len() {
                            if a_req {
                                a.grad[i] += out.grad[i];
                            }
                            if b_req {
                                b.grad[i] += out.grad[i];
                            }
                        }
                    }
                }

                Operation::Mul => {
                    let a_id = node.input0;
                    let b_id = node.input1.expect("Mul op must have 2 inputs");

                    let a_req = store.get(a_id).requires_grad;
                    let b_req = store.get(b_id).requires_grad;
                    if !a_req && !b_req {
                        continue;
                    }

                    if a_req && a_id < active.len() {
                        active[a_id] = true;
                    }
                    if b_req && b_id < active.len() {
                        active[b_id] = true;
                    }

                    if a_id == b_id {
                        if a_req {
                            let (a, out) = store.get_mut_and_1(a_id, node.output);
                            assert_eq!(
                                a.grad.len(),
                                a.data.len(),
                                "Mul backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                                a_id
                            );
                            debug_assert_eq!(
                                out.grad.len(),
                                a.grad.len(),
                                "Mul backward: grad size mismatch"
                            );
                            for i in 0..out.grad.len() {
                                a.grad[i] += 2.0 * a.data[i] * out.grad[i];
                            }
                        }
                    } else {
                        let (a, b, out) = store.get2_mut_and_1(a_id, b_id, node.output);
                        if a_req {
                            assert_eq!(
                                a.grad.len(),
                                a.data.len(),
                                "Mul backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                                a_id
                            );
                        }
                        if b_req {
                            assert_eq!(
                                b.grad.len(),
                                b.data.len(),
                                "Mul backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                                b_id
                            );
                        }
                        debug_assert_eq!(
                            out.grad.len(),
                            a.grad.len(),
                            "Mul backward: grad size mismatch"
                        );
                        debug_assert_eq!(
                            out.grad.len(),
                            b.grad.len(),
                            "Mul backward: grad size mismatch"
                        );
                        for i in 0..out.grad.len() {
                            if a_req {
                                a.grad[i] += b.data[i] * out.grad[i];
                            }
                            if b_req {
                                b.grad[i] += a.data[i] * out.grad[i];
                            }
                        }
                    }
                }

                Operation::MatMul => {
                    let a_id = node.input0;
                    let b_id = node.input1.expect("MatMul op must have 2 inputs");

                    let a_req = store.get(a_id).requires_grad;
                    let b_req = store.get(b_id).requires_grad;
                    if !a_req && !b_req {
                        continue;
                    }

                    if a_req && a_id < active.len() {
                        active[a_id] = true;
                    }
                    if b_req && b_id < active.len() {
                        active[b_id] = true;
                    }

                    // Same-id case (X @ X): dX = dC @ X^T + X^T @ dC
                    if a_id == b_id {
                        let (a, out) = store.get_mut_and_1(a_id, node.output);
                        assert_eq!(a.shape.len(), 2, "A must be 2D for MatMul backward");
                        assert_eq!(
                            a.grad.len(),
                            a.data.len(),
                            "MatMul backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                            a_id
                        );
                        let (m, n) = (a.shape[0], a.shape[1]);
                        assert_eq!(m, n, "MatMul backward (same-id): requires square matrix");
                        let p = n;
                        assert_eq!(out.grad.len(), m * p, "Output grad has wrong size");

                        if a_req {
                            let kernel = *matmul_kernel_cache
                                .entry((m, n, p))
                                .or_insert_with(|| select_kernel(m.max(n).max(p)));
                            let d_c = out.grad.as_slice();

                            // Term 1: dC @ A^T
                            match kernel {
                                KernelType::Naive => matmul_naive_add_into_slices_b_transposed(
                                    d_c,
                                    m,
                                    p,
                                    &a.data,
                                    n,
                                    &mut a.grad,
                                ),
                                KernelType::Tiled => matmul_tiled_add_into_slices_b_transposed(
                                    d_c,
                                    m,
                                    p,
                                    &a.data,
                                    n,
                                    &mut a.grad,
                                    16,
                                ),
                                KernelType::TiledMP => {
                                    matmul_tiled_mp_add_into_slices_b_transposed(
                                        d_c,
                                        m,
                                        p,
                                        &a.data,
                                        n,
                                        &mut a.grad,
                                        16,
                                    )
                                }
                            }

                            // Term 2: A^T @ dC
                            match kernel {
                                KernelType::Naive => matmul_naive_add_into_slices_a_transposed(
                                    &a.data,
                                    m,
                                    n,
                                    d_c,
                                    p,
                                    &mut a.grad,
                                ),
                                KernelType::Tiled => matmul_tiled_add_into_slices_a_transposed(
                                    &a.data,
                                    m,
                                    n,
                                    d_c,
                                    p,
                                    &mut a.grad,
                                    16,
                                ),
                                KernelType::TiledMP => {
                                    matmul_tiled_mp_add_into_slices_a_transposed(
                                        &a.data,
                                        m,
                                        n,
                                        d_c,
                                        p,
                                        &mut a.grad,
                                        16,
                                    )
                                }
                            }
                        }

                        continue;
                    }

                    let (a, b, out) = store.get2_mut_and_1(a_id, b_id, node.output);
                    assert_eq!(a.shape.len(), 2, "A must be 2D for MatMul backward");
                    assert_eq!(b.shape.len(), 2, "B must be 2D for MatMul backward");
                    assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match");

                    if a_req {
                        assert_eq!(
                            a.grad.len(),
                            a.data.len(),
                            "MatMul backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                            a_id
                        );
                    }
                    if b_req {
                        assert_eq!(
                            b.grad.len(),
                            b.data.len(),
                            "MatMul backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                            b_id
                        );
                    }

                    let (m, n) = (a.shape[0], a.shape[1]);
                    let p = b.shape[1];
                    assert_eq!(out.grad.len(), m * p, "Output grad has wrong size");

                    let kernel = *matmul_kernel_cache
                        .entry((m, n, p))
                        .or_insert_with(|| select_kernel(m.max(n).max(p)));
                    let d_c = out.grad.as_slice();

                    if a_req {
                        // dA = dC @ B^T
                        match kernel {
                            KernelType::Naive => matmul_naive_add_into_slices_b_transposed(
                                d_c,
                                m,
                                p,
                                &b.data,
                                n,
                                &mut a.grad,
                            ),
                            KernelType::Tiled => matmul_tiled_add_into_slices_b_transposed(
                                d_c,
                                m,
                                p,
                                &b.data,
                                n,
                                &mut a.grad,
                                16,
                            ),
                            KernelType::TiledMP => matmul_tiled_mp_add_into_slices_b_transposed(
                                d_c,
                                m,
                                p,
                                &b.data,
                                n,
                                &mut a.grad,
                                16,
                            ),
                        }
                    }

                    if b_req {
                        // dB = A^T @ dC
                        match kernel {
                            KernelType::Naive => matmul_naive_add_into_slices_a_transposed(
                                &a.data,
                                m,
                                n,
                                d_c,
                                p,
                                &mut b.grad,
                            ),
                            KernelType::Tiled => matmul_tiled_add_into_slices_a_transposed(
                                &a.data,
                                m,
                                n,
                                d_c,
                                p,
                                &mut b.grad,
                                16,
                            ),
                            KernelType::TiledMP => matmul_tiled_mp_add_into_slices_a_transposed(
                                &a.data,
                                m,
                                n,
                                d_c,
                                p,
                                &mut b.grad,
                                16,
                            ),
                        }
                    }
                }

                Operation::ReLU => {
                    let input_id = node.input0;
                    if !store.get(input_id).requires_grad {
                        continue;
                    }
                    if input_id < active.len() {
                        active[input_id] = true;
                    }

                    let (input, out) = store.get_mut_and_1(input_id, node.output);
                    assert_eq!(
                        input.grad.len(),
                        input.data.len(),
                        "ReLU backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                        input_id
                    );
                    debug_assert_eq!(
                        out.grad.len(),
                        input.grad.len(),
                        "ReLU backward: grad size mismatch"
                    );
                    for i in 0..out.grad.len() {
                        if input.data[i] > 0.0 {
                            input.grad[i] += out.grad[i];
                        }
                    }
                }

                Operation::MSE => {
                    let pred_id = node.input0;
                    let target_id = node.input1.expect("MSE op must have 2 inputs");

                    let pred_req = store.get(pred_id).requires_grad;
                    let target_req = store.get(target_id).requires_grad;
                    if pred_id == target_id {
                        continue;
                    }

                    if pred_req && pred_id < active.len() {
                        active[pred_id] = true;
                    }
                    if target_req && target_id < active.len() {
                        active[target_id] = true;
                    }

                    let (pred, target, out) = store.get2_mut_and_1(pred_id, target_id, node.output);
                    assert_eq!(
                        out.grad.len(),
                        1,
                        "MSE backward: loss output must be scalar (grad len == 1)"
                    );
                    let upstream = out.grad[0];

                    assert_eq!(
                        pred.data.len(),
                        target.data.len(),
                        "MSE backward: pred and target must have same length"
                    );
                    assert!(
                        !pred.data.is_empty(),
                        "MSE backward: empty tensors are not supported"
                    );
                    if pred_req {
                        assert_eq!(
                            pred.grad.len(),
                            pred.data.len(),
                            "MSE backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                            pred_id
                        );
                    }
                    if target_req {
                        assert_eq!(
                            target.grad.len(),
                            target.data.len(),
                            "MSE backward: missing/invalid grad buffer for tensor {} (use backward_zeroed if grads were released)",
                            target_id
                        );
                    }
                    let n = pred.data.len() as f32;
                    let scale = 2.0 / n;

                    for i in 0..pred.data.len() {
                        let diff = pred.data[i] - target.data[i];
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

#[cfg(test)]
mod tests {
    use super::Graph;
    use crate::mem::pool::MemoryPool;
    use crate::tensor::store::TensorStore;
    use crate::tensor::tensor::{Tensor, matmul, relu};

    #[test]
    fn matmul_backward_matches_sums_when_loss_grad_is_ones() {
        let mut store = TensorStore::new();
        let mut graph = Graph::new();
        let mut pool = MemoryPool::new();
        pool.enabled = true;

        // A: 2x3, B: 3x2
        let a_id = store.add(Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            true,
        ));
        let b_id = store.add(Tensor::new(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 2],
            true,
        ));

        let out_id = matmul(a_id, b_id, &mut store, &mut graph);
        let seed = vec![1.0; store.get(out_id).grad.len()];
        graph.backward_seeded(&mut store, out_id, &seed, &mut pool);

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
        let mut pool = MemoryPool::new();
        pool.enabled = true;

        let x_id = store.add(Tensor::new(vec![-1.0, 2.0, 0.5, 0.0], vec![4], true));
        let y_id = relu(x_id, &mut store, &mut graph);

        let seed = vec![1.0; store.get(y_id).grad.len()];
        graph.backward_seeded(&mut store, y_id, &seed, &mut pool);

        // dy/dx is 0 for x<=0, 1 for x>0 (with upstream grad = 1)
        assert_eq!(store.get(x_id).grad, vec![0.0, 1.0, 1.0, 0.0]);
    }
}
