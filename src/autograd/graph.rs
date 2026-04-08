use crate::autograd::node::{Node, Operation};
use crate::memory::pool::MemoryPool;
use crate::planner::grad_planner::GradPlanner;
use crate::tensor::store::TensorStore;

use crate::kernels::selector::{
    KernelType, matmul_add_into_slices_a_transposed, matmul_add_into_slices_b_transposed,
    select_kernel_mm,
};

use std::collections::HashMap;

trait GradAllocator {
    fn ensure_grad(&mut self, store: &mut TensorStore, id: usize);
    fn seed_grad(&mut self, store: &mut TensorStore, id: usize, seed: &[f32]);
    fn release_if_intermediate(&mut self, store: &mut TensorStore, id: usize);
}

struct PoolGradAllocator<'a> {
    pool: &'a mut MemoryPool,
}

impl<'a> GradAllocator for PoolGradAllocator<'a> {
    #[inline]
    fn ensure_grad(&mut self, store: &mut TensorStore, id: usize) {
        let t = store.get_mut(id);
        if !t.requires_grad {
            return;
        }

        if t.grad.is_empty() {
            t.grad = self.pool.get(t.data.len());
        } else {
            debug_assert_eq!(
                t.grad.len(),
                t.data.len(),
                "Tensor {}: grad len != data len",
                id
            );
        }
    }

    #[inline]
    fn seed_grad(&mut self, store: &mut TensorStore, id: usize, seed: &[f32]) {
        self.ensure_grad(store, id);
        let t = store.get_mut(id);
        if !t.requires_grad {
            return;
        }
        assert_eq!(
            t.data.len(),
            seed.len(),
            "seed length must match tensor size"
        );
        assert_eq!(t.grad.len(), seed.len(), "missing/invalid grad buffer");
        t.grad.copy_from_slice(seed);
    }

    #[inline]
    fn release_if_intermediate(&mut self, store: &mut TensorStore, id: usize) {
        let t = store.get_mut(id);
        if t.creator.is_some() && !t.grad.is_empty() {
            self.pool.release(std::mem::take(&mut t.grad));
        }
    }
}

struct PlannedGradAllocator {
    planner: GradPlanner,
}

impl PlannedGradAllocator {
    fn new(planner: GradPlanner) -> Self {
        Self { planner }
    }
}

impl GradAllocator for PlannedGradAllocator {
    #[inline]
    fn ensure_grad(&mut self, store: &mut TensorStore, id: usize) {
        let t = store.get_mut(id);
        if !t.requires_grad {
            return;
        }
        if t.grad.is_empty() {
            let len = t.data.len();
            t.grad = self.planner.checkout(id, len);
        } else {
            debug_assert_eq!(t.grad.len(), t.data.len());
        }
    }

    #[inline]
    fn seed_grad(&mut self, store: &mut TensorStore, id: usize, seed: &[f32]) {
        self.ensure_grad(store, id);
        let t = store.get_mut(id);
        if !t.requires_grad {
            return;
        }
        assert_eq!(
            t.data.len(),
            seed.len(),
            "seed length must match tensor size"
        );
        assert_eq!(t.grad.len(), seed.len(), "missing/invalid grad buffer");
        t.grad.copy_from_slice(seed);
    }

    #[inline]
    fn release_if_intermediate(&mut self, store: &mut TensorStore, id: usize) {
        let t = store.get_mut(id);
        if t.creator.is_some() && !t.grad.is_empty() {
            let buf = std::mem::take(&mut t.grad);
            self.planner.checkin(id, buf);
        }
    }
}

/// A minimal reverse-mode autograd tape.
///
/// Node ids are stable indices into `nodes`.
pub struct Graph {
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: Node) -> usize {
        let node_id = self.nodes.len();
        self.nodes.push(node);
        node_id
    }

    pub fn backward(&self, store: &mut TensorStore, loss_id: usize, pool: &mut MemoryPool) {
        // Contract: `backward(loss_id)` seeds scalar loss with 1.0.
        // For non-scalar outputs, use `backward_seeded`.
        let mut alloc = PoolGradAllocator { pool };

        if store.get(loss_id).requires_grad {
            assert_eq!(
                store.get(loss_id).data.len(),
                1,
                "Graph::backward(loss_id): loss must be scalar; use backward_seeded for non-scalar"
            );
            alloc.seed_grad(store, loss_id, &[1.0]);
        }

        self.backward_internal(store, loss_id, &mut alloc);
        alloc.release_if_intermediate(store, loss_id);
    }

    pub fn backward_seeded(
        &self,
        store: &mut TensorStore,
        output_id: usize,
        seed: &[f32],
        pool: &mut MemoryPool,
    ) {
        let mut alloc = PoolGradAllocator { pool };
        if store.get(output_id).requires_grad {
            alloc.seed_grad(store, output_id, seed);
        }

        self.backward_internal(store, output_id, &mut alloc);
        alloc.release_if_intermediate(store, output_id);
    }

    /// Research-grade deterministic backward pass.
    ///
    /// This builds a backward liveness-based `GradPlanner` and uses it to allocate and
    /// recycle gradient buffers deterministically (no `MemoryPool::get/release`).
    pub fn backward_planned(&self, store: &mut TensorStore, loss_id: usize) -> GradPlanner {
        if store.get(loss_id).requires_grad {
            assert_eq!(
                store.get(loss_id).data.len(),
                1,
                "Graph::backward_planned(loss_id): loss must be scalar; use backward_seeded_planned for non-scalar"
            );
        }

        let planner = GradPlanner::build(self, store, loss_id);
        let mut alloc = PlannedGradAllocator::new(planner);
        if store.get(loss_id).requires_grad {
            alloc.seed_grad(store, loss_id, &[1.0]);
        }

        self.backward_internal(store, loss_id, &mut alloc);
        alloc.release_if_intermediate(store, loss_id);
        alloc.planner
    }

    pub fn backward_seeded_planned(
        &self,
        store: &mut TensorStore,
        output_id: usize,
        seed: &[f32],
    ) -> GradPlanner {
        let planner = GradPlanner::build(self, store, output_id);
        let mut alloc = PlannedGradAllocator::new(planner);
        if store.get(output_id).requires_grad {
            alloc.seed_grad(store, output_id, seed);
        }

        self.backward_internal(store, output_id, &mut alloc);
        alloc.release_if_intermediate(store, output_id);
        alloc.planner
    }

    fn backward_internal(
        &self,
        store: &mut TensorStore,
        output_id: usize,
        alloc: &mut dyn GradAllocator,
    ) {
        // Track which tensors have an upstream gradient path from `output_id`.
        let mut active: Vec<bool> = vec![false; store.tensors.len()];
        if output_id < active.len() && store.get(output_id).requires_grad {
            active[output_id] = true;
        }

        // Cache kernel selection for repeated MatMul shapes within this backward pass.
        // Key: (m, k, n) where output is (m x n) and reduction is k.
        let mut matmul_kernel_cache: HashMap<(usize, usize, usize), KernelType> = HashMap::new();

        for node in self.nodes.iter().rev() {
            if node.output >= active.len() || !active[node.output] {
                continue;
            }

            // Ensure the upstream gradient buffer exists.
            alloc.ensure_grad(store, node.output);

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

                    if a_req {
                        alloc.ensure_grad(store, a_id);
                    }
                    if b_req {
                        alloc.ensure_grad(store, b_id);
                    }

                    if a_id == b_id {
                        if a_req {
                            let (a, out) = store.get_mut_and_1(a_id, node.output);
                            assert_eq!(
                                a.grad.len(),
                                a.data.len(),
                                "Add backward: missing/invalid grad buffer for tensor {}",
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
                                "Add backward: missing/invalid grad buffer for tensor {}",
                                a_id
                            );
                        }
                        if b_req {
                            assert_eq!(
                                b.grad.len(),
                                b.data.len(),
                                "Add backward: missing/invalid grad buffer for tensor {}",
                                b_id
                            );
                        }
                        if a_req {
                            debug_assert_eq!(
                                out.grad.len(),
                                a.grad.len(),
                                "Add backward: grad size mismatch"
                            );
                        }
                        if b_req {
                            debug_assert_eq!(
                                out.grad.len(),
                                b.grad.len(),
                                "Add backward: grad size mismatch"
                            );
                        }
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

                    if a_req {
                        alloc.ensure_grad(store, a_id);
                    }
                    if b_req {
                        alloc.ensure_grad(store, b_id);
                    }

                    if a_id == b_id {
                        if a_req {
                            let (a, out) = store.get_mut_and_1(a_id, node.output);
                            assert_eq!(
                                a.grad.len(),
                                a.data.len(),
                                "Mul backward: missing/invalid grad buffer for tensor {}",
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
                                "Mul backward: missing/invalid grad buffer for tensor {}",
                                a_id
                            );
                        }
                        if b_req {
                            assert_eq!(
                                b.grad.len(),
                                b.data.len(),
                                "Mul backward: missing/invalid grad buffer for tensor {}",
                                b_id
                            );
                        }
                        if a_req {
                            debug_assert_eq!(
                                out.grad.len(),
                                a.grad.len(),
                                "Mul backward: grad size mismatch"
                            );
                        }
                        if b_req {
                            debug_assert_eq!(
                                out.grad.len(),
                                b.grad.len(),
                                "Mul backward: grad size mismatch"
                            );
                        }
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

                    if a_req {
                        alloc.ensure_grad(store, a_id);
                    }
                    if b_req {
                        alloc.ensure_grad(store, b_id);
                    }

                    // Same-id case (X @ X): dX = dC @ X^T + X^T @ dC
                    if a_id == b_id {
                        let (a, out) = store.get_mut_and_1(a_id, node.output);
                        assert_eq!(a.shape.len(), 2, "A must be 2D for MatMul backward");
                        assert_eq!(
                            a.grad.len(),
                            a.data.len(),
                            "MatMul backward: missing/invalid grad buffer for tensor {}",
                            a_id
                        );
                        let (m, n) = (a.shape[0], a.shape[1]);
                        assert_eq!(m, n, "MatMul backward (same-id): requires square matrix");
                        let p = n;
                        assert_eq!(out.grad.len(), m * p, "Output grad has wrong size");

                        if a_req {
                            let mut get_kernel = |mm: usize, kk: usize, nn: usize| {
                                *matmul_kernel_cache
                                    .entry((mm, kk, nn))
                                    .or_insert_with(|| select_kernel_mm(mm, kk, nn))
                            };
                            let d_c = out.grad.as_slice();

                            // Term 1: dC @ A^T
                            let k1 = get_kernel(m, p, n);
                            matmul_add_into_slices_b_transposed(
                                k1,
                                d_c,
                                m,
                                p,
                                &a.data,
                                n,
                                &mut a.grad,
                            );

                            // Term 2: A^T @ dC
                            let k2 = get_kernel(n, m, p);
                            matmul_add_into_slices_a_transposed(
                                k2,
                                &a.data,
                                m,
                                n,
                                d_c,
                                p,
                                &mut a.grad,
                            );
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
                            "MatMul backward: missing/invalid grad buffer for tensor {}",
                            a_id
                        );
                    }
                    if b_req {
                        assert_eq!(
                            b.grad.len(),
                            b.data.len(),
                            "MatMul backward: missing/invalid grad buffer for tensor {}",
                            b_id
                        );
                    }

                    let (m, n) = (a.shape[0], a.shape[1]);
                    let p = b.shape[1];
                    assert_eq!(out.grad.len(), m * p, "Output grad has wrong size");

                    let d_c = out.grad.as_slice();

                    let mut get_kernel = |mm: usize, kk: usize, nn: usize| {
                        *matmul_kernel_cache
                            .entry((mm, kk, nn))
                            .or_insert_with(|| select_kernel_mm(mm, kk, nn))
                    };

                    if a_req {
                        // dA = dC @ B^T
                        let k_da = get_kernel(m, p, n);
                        matmul_add_into_slices_b_transposed(
                            k_da,
                            d_c,
                            m,
                            p,
                            &b.data,
                            n,
                            &mut a.grad,
                        );
                    }

                    if b_req {
                        // dB = A^T @ dC
                        let k_db = get_kernel(n, m, p);
                        matmul_add_into_slices_a_transposed(
                            k_db,
                            &a.data,
                            m,
                            n,
                            d_c,
                            p,
                            &mut b.grad,
                        );
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

                    alloc.ensure_grad(store, input_id);

                    let (input, out) = store.get_mut_and_1(input_id, node.output);
                    assert_eq!(
                        input.grad.len(),
                        input.data.len(),
                        "ReLU backward: missing/invalid grad buffer for tensor {}",
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

                    if pred_req {
                        alloc.ensure_grad(store, pred_id);
                    }
                    if target_req {
                        alloc.ensure_grad(store, target_id);
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
                            "MSE backward: missing/invalid grad buffer for tensor {}",
                            pred_id
                        );
                    }
                    if target_req {
                        assert_eq!(
                            target.grad.len(),
                            target.data.len(),
                            "MSE backward: missing/invalid grad buffer for tensor {}",
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

            // Release intermediate output gradient buffers as soon as they are no longer needed.
            // After a node is processed in reverse, its output grad won't be used again.
            if node.output != output_id {
                alloc.release_if_intermediate(store, node.output);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Graph;
    use crate::memory::pool::MemoryPool;
    use crate::tensor::store::TensorStore;
    use crate::tensor::tensor::{Tensor, add, matmul, mul, relu};

    fn loss_sum(output: &[f32]) -> f32 {
        output.iter().copied().sum::<f32>()
    }

    fn finite_diff_grad(mut x: Vec<f32>, eps: f32, f: impl Fn(&[f32]) -> f32) -> Vec<f32> {
        let mut g = vec![0.0; x.len()];
        for i in 0..x.len() {
            let orig = x[i];
            x[i] = orig + eps;
            let f_pos = f(&x);
            x[i] = orig - eps;
            let f_neg = f(&x);
            x[i] = orig;
            g[i] = (f_pos - f_neg) / (2.0 * eps);
        }
        g
    }

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
        let seed = vec![1.0; store.get(out_id).data.len()];
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

        let seed = vec![1.0; store.get(y_id).data.len()];
        graph.backward_seeded(&mut store, y_id, &seed, &mut pool);

        // dy/dx is 0 for x<=0, 1 for x>0 (with upstream grad = 1)
        assert_eq!(store.get(x_id).grad, vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn finite_diff_add_grad_matches_backward() {
        let eps = 1e-3f32;
        let tol = 2e-2f32;

        let a0 = vec![0.2, -1.1, 0.7, 1.3, -0.4, 2.0];
        let b0 = vec![1.0, 0.5, -0.3, 0.2, 0.9, -1.7];

        // Analytic grad via autograd
        let mut store = TensorStore::new();
        let mut graph = Graph::new();
        let mut pool = MemoryPool::new();
        pool.enabled = true;

        let a_id = store.add(Tensor::new(a0.clone(), vec![2, 3], true));
        let b_id = store.add(Tensor::new(b0.clone(), vec![2, 3], true));
        let out_id = add(a_id, b_id, &mut store, &mut graph);

        let seed = vec![1.0; store.get(out_id).data.len()];
        graph.backward_seeded(&mut store, out_id, &seed, &mut pool);

        let ga = store.get(a_id).grad.clone();
        let gb = store.get(b_id).grad.clone();

        // Numeric grad (loss = sum(out))
        let f_a = |a: &[f32]| {
            let mut store = TensorStore::new();
            let mut graph = Graph::new();
            let a_id = store.add(Tensor::new(a.to_vec(), vec![2, 3], true));
            let b_id = store.add(Tensor::new(b0.clone(), vec![2, 3], true));
            let out_id = add(a_id, b_id, &mut store, &mut graph);
            loss_sum(&store.get(out_id).data)
        };
        let f_b = |b: &[f32]| {
            let mut store = TensorStore::new();
            let mut graph = Graph::new();
            let a_id = store.add(Tensor::new(a0.clone(), vec![2, 3], true));
            let b_id = store.add(Tensor::new(b.to_vec(), vec![2, 3], true));
            let out_id = add(a_id, b_id, &mut store, &mut graph);
            loss_sum(&store.get(out_id).data)
        };

        let nga = finite_diff_grad(a0.clone(), eps, f_a);
        let ngb = finite_diff_grad(b0.clone(), eps, f_b);

        for i in 0..ga.len() {
            assert!(
                (ga[i] - nga[i]).abs() < tol,
                "dA[{}]: {} vs {}",
                i,
                ga[i],
                nga[i]
            );
            assert!(
                (gb[i] - ngb[i]).abs() < tol,
                "dB[{}]: {} vs {}",
                i,
                gb[i],
                ngb[i]
            );
        }
    }

    #[test]
    fn finite_diff_mul_grad_matches_backward() {
        let eps = 1e-3f32;
        let tol = 2e-2f32;

        let a0 = vec![0.2, -1.1, 0.7, 1.3, -0.4, 2.0];
        let b0 = vec![1.0, 0.5, -0.3, 0.2, 0.9, -1.7];

        // Analytic grad via autograd
        let mut store = TensorStore::new();
        let mut graph = Graph::new();
        let mut pool = MemoryPool::new();
        pool.enabled = true;

        let a_id = store.add(Tensor::new(a0.clone(), vec![2, 3], true));
        let b_id = store.add(Tensor::new(b0.clone(), vec![2, 3], true));
        let out_id = mul(a_id, b_id, &mut store, &mut graph);

        let seed = vec![1.0; store.get(out_id).data.len()];
        graph.backward_seeded(&mut store, out_id, &seed, &mut pool);

        let ga = store.get(a_id).grad.clone();
        let gb = store.get(b_id).grad.clone();

        // Numeric grad (loss = sum(out))
        let f_a = |a: &[f32]| {
            let mut store = TensorStore::new();
            let mut graph = Graph::new();
            let a_id = store.add(Tensor::new(a.to_vec(), vec![2, 3], true));
            let b_id = store.add(Tensor::new(b0.clone(), vec![2, 3], true));
            let out_id = mul(a_id, b_id, &mut store, &mut graph);
            loss_sum(&store.get(out_id).data)
        };
        let f_b = |b: &[f32]| {
            let mut store = TensorStore::new();
            let mut graph = Graph::new();
            let a_id = store.add(Tensor::new(a0.clone(), vec![2, 3], true));
            let b_id = store.add(Tensor::new(b.to_vec(), vec![2, 3], true));
            let out_id = mul(a_id, b_id, &mut store, &mut graph);
            loss_sum(&store.get(out_id).data)
        };

        let nga = finite_diff_grad(a0.clone(), eps, f_a);
        let ngb = finite_diff_grad(b0.clone(), eps, f_b);

        for i in 0..ga.len() {
            assert!(
                (ga[i] - nga[i]).abs() < tol,
                "dA[{}]: {} vs {}",
                i,
                ga[i],
                nga[i]
            );
            assert!(
                (gb[i] - ngb[i]).abs() < tol,
                "dB[{}]: {} vs {}",
                i,
                gb[i],
                ngb[i]
            );
        }
    }

    #[test]
    fn finite_diff_matmul_rectangular_grad_matches_backward() {
        let eps = 1e-3f32;
        let tol = 5e-2f32;

        // A: 2x3, B: 3x4 => C: 2x4
        let a0 = vec![0.5, -0.2, 1.1, 0.7, 0.3, -0.9];
        let b0 = vec![
            0.4, -1.2, 0.8, 0.1, 0.7, 0.3, -0.5, 1.0, -0.6, 0.2, 0.9, -0.4,
        ];

        // Analytic grad via autograd
        let mut store = TensorStore::new();
        let mut graph = Graph::new();
        let mut pool = MemoryPool::new();
        pool.enabled = true;

        let a_id = store.add(Tensor::new(a0.clone(), vec![2, 3], true));
        let b_id = store.add(Tensor::new(b0.clone(), vec![3, 4], true));
        let out_id = matmul(a_id, b_id, &mut store, &mut graph);

        let seed = vec![1.0; store.get(out_id).data.len()];
        graph.backward_seeded(&mut store, out_id, &seed, &mut pool);

        let ga = store.get(a_id).grad.clone();
        let gb = store.get(b_id).grad.clone();

        // Numeric grad (loss = sum(out))
        let f_a = |a: &[f32]| {
            let mut store = TensorStore::new();
            let mut graph = Graph::new();
            let a_id = store.add(Tensor::new(a.to_vec(), vec![2, 3], true));
            let b_id = store.add(Tensor::new(b0.clone(), vec![3, 4], true));
            let out_id = matmul(a_id, b_id, &mut store, &mut graph);
            loss_sum(&store.get(out_id).data)
        };
        let f_b = |b: &[f32]| {
            let mut store = TensorStore::new();
            let mut graph = Graph::new();
            let a_id = store.add(Tensor::new(a0.clone(), vec![2, 3], true));
            let b_id = store.add(Tensor::new(b.to_vec(), vec![3, 4], true));
            let out_id = matmul(a_id, b_id, &mut store, &mut graph);
            loss_sum(&store.get(out_id).data)
        };

        let nga = finite_diff_grad(a0.clone(), eps, f_a);
        let ngb = finite_diff_grad(b0.clone(), eps, f_b);

        for i in 0..ga.len() {
            assert!(
                (ga[i] - nga[i]).abs() < tol,
                "dA[{}]: {} vs {}",
                i,
                ga[i],
                nga[i]
            );
        }
        for i in 0..gb.len() {
            assert!(
                (gb[i] - ngb[i]).abs() < tol,
                "dB[{}]: {} vs {}",
                i,
                gb[i],
                ngb[i]
            );
        }
    }

    #[test]
    fn finite_diff_relu_grad_matches_backward() {
        let eps = 1e-3f32;
        let tol = 2e-2f32;

        let x0 = vec![-1.2, 0.7, 2.3, -0.4];

        // Analytic grad via autograd
        let mut store = TensorStore::new();
        let mut graph = Graph::new();
        let mut pool = MemoryPool::new();
        pool.enabled = true;

        let x_id = store.add(Tensor::new(x0.clone(), vec![4], true));
        let out_id = relu(x_id, &mut store, &mut graph);
        let seed = vec![1.0; store.get(out_id).data.len()];
        graph.backward_seeded(&mut store, out_id, &seed, &mut pool);
        let gx = store.get(x_id).grad.clone();

        // Numeric grad (loss = sum(out))
        let f_x = |x: &[f32]| {
            let mut store = TensorStore::new();
            let mut graph = Graph::new();
            let x_id = store.add(Tensor::new(x.to_vec(), vec![4], true));
            let out_id = relu(x_id, &mut store, &mut graph);
            loss_sum(&store.get(out_id).data)
        };
        let ngx = finite_diff_grad(x0, eps, f_x);

        for i in 0..gx.len() {
            assert!(
                (gx[i] - ngx[i]).abs() < tol,
                "dX[{}]: {} vs {}",
                i,
                gx[i],
                ngx[i]
            );
        }
    }

    #[test]
    fn finite_diff_reuse_tensor_multiple_times_in_graph() {
        let eps = 1e-3f32;
        let tol = 5e-2f32;

        // y = x * x + x, loss = sum(y)
        let x0 = vec![0.3, -0.8, 1.5, 0.2];

        // Analytic grad via autograd
        let mut store = TensorStore::new();
        let mut graph = Graph::new();
        let mut pool = MemoryPool::new();
        pool.enabled = true;

        let x_id = store.add(Tensor::new(x0.clone(), vec![4], true));
        let y_id = mul(x_id, x_id, &mut store, &mut graph);
        let z_id = add(y_id, x_id, &mut store, &mut graph);
        let seed = vec![1.0; store.get(z_id).data.len()];
        graph.backward_seeded(&mut store, z_id, &seed, &mut pool);
        let gx = store.get(x_id).grad.clone();

        // Numeric grad
        let f_x = |x: &[f32]| {
            let mut store = TensorStore::new();
            let mut graph = Graph::new();
            let x_id = store.add(Tensor::new(x.to_vec(), vec![4], true));
            let y_id = mul(x_id, x_id, &mut store, &mut graph);
            let z_id = add(y_id, x_id, &mut store, &mut graph);
            loss_sum(&store.get(z_id).data)
        };
        let ngx = finite_diff_grad(x0, eps, f_x);

        for i in 0..gx.len() {
            assert!(
                (gx[i] - ngx[i]).abs() < tol,
                "dX[{}]: {} vs {}",
                i,
                gx[i],
                ngx[i]
            );
        }
    }
}
