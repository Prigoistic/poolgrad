use crate::autograd::node::{Node, Operation};
use crate::tensor::store::TensorStore;
use crate::mem::pool::MemoryPool;
use crate::kernels::naive::matmul_naive_add_into_slices;
use crate::kernels::tiled::matmul_tiled_into_slices;
use crate::kernels::tiled_mp::matmul_tiled_mp_into_slices;
use crate::kernels::selector::{select_kernel, KernelType};
use rayon::prelude::*;

fn parallel_enabled() -> bool {
    std::env::var("POOLGRAD_PAR")
        .ok()
        .as_deref()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
}

fn par_min_elems() -> usize {
    std::env::var("POOLGRAD_PAR_MIN_ELEMS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16 * 1024)
}

fn transpose_into(src: &[f32], src_rows: usize, src_cols: usize, dst: &mut [f32]) {
    assert_eq!(dst.len(), src_rows * src_cols);

    if parallel_enabled() && src_rows * src_cols >= par_min_elems() {
        // dst is (src_cols x src_rows) row-major, i.e. each dst row corresponds to a src column.
        dst.par_chunks_mut(src_rows)
            .enumerate()
            .for_each(|(c, dst_row)| {
                for r in 0..src_rows {
                    dst_row[r] = src[r * src_cols + c];
                }
            });
    } else {
        for r in 0..src_rows {
            for c in 0..src_cols {
                dst[c * src_rows + r] = src[r * src_cols + c];
            }
        }
    }
}



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

    pub fn backward(&self, store: &mut TensorStore, loss_id: usize, pool: &mut MemoryPool) {
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
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];

                    let a_req = store.get(a_id).requires_grad;
                    let b_req = store.get(b_id).requires_grad;

                    if !a_req && !b_req {
                        continue;
                    }

                    if a_id == b_id {
                        if a_req {
                            let (a, out) = store.get_mut_and_1(a_id, node.output);
                            for i in 0..out.grad.len() {
                                a.grad[i] += 2.0 * out.grad[i];
                            }
                        }
                    } else {
                        let (a, b, out) = store.get2_mut_and_1(a_id, b_id, node.output);
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
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];

                    let a_req = store.get(a_id).requires_grad;
                    let b_req = store.get(b_id).requires_grad;

                    if !a_req && !b_req {
                        continue;
                    }

                    if a_id == b_id {
                        // y = a * a => dy/da = 2a
                        if a_req {
                            let (a, out) = store.get_mut_and_1(a_id, node.output);
                            for i in 0..out.grad.len() {
                                a.grad[i] += 2.0 * a.data[i] * out.grad[i];
                            }
                        }
                    } else {
                        let (a, b, out) = store.get2_mut_and_1(a_id, b_id, node.output);
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
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let a_req = store.get(a_id).requires_grad;
                    let b_req = store.get(b_id).requires_grad;

                    if !a_req && !b_req {
                        continue;
                    }

                    // dA = dC @ B^T, dB = A^T @ dC
                    // We compute this using the same kernel dispatch as forward. Since gradients
                    // must accumulate (a tensor can be used multiple times), all kernel paths
                    // used here add into the provided output buffer.

                    // Same-id case is rare and shape-sensitive (dA and dB have different shapes
                    // in general). Keep the simple loop implementation as a safe fallback.
                    if a_id == b_id {
                        // Same-id case: implies square matrix multiply in forward (n == m).
                        let (a, out) = store.get_mut_and_1(a_id, node.output);

                        assert_eq!(a.shape.len(), 2, "A must be 2D for MatMul backward");
                        let (m, n) = (a.shape[0], a.shape[1]);
                        assert_eq!(m, n, "MatMul backward (same-id): requires square matrix");
                        let p = n;
                        assert_eq!(out.grad.len(), m * p, "Output grad has wrong size");

                        if a_req {
                            let mut d_a = pool.get(m * n);
                            let mut d_b = pool.get(n * p);

                            for i in 0..m {
                                for j in 0..p {
                                    let go = out.grad[i * p + j];
                                    for k in 0..n {
                                        d_a[i * n + k] += go * a.data[k * p + j];
                                        d_b[k * p + j] += a.data[i * n + k] * go;
                                    }
                                }
                            }

                            for idx in 0..a.grad.len() {
                                a.grad[idx] += d_a[idx];
                                if b_req && idx < d_b.len() {
                                    a.grad[idx] += d_b[idx];
                                }
                            }

                            pool.release(d_a);
                            pool.release(d_b);
                        }

                        continue;
                    }

                    let (a, b, out) = store.get2_mut_and_1(a_id, b_id, node.output);

                    assert_eq!(a.shape.len(), 2, "A must be 2D for MatMul backward");
                    assert_eq!(b.shape.len(), 2, "B must be 2D for MatMul backward");
                    assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match");

                    let (m, n) = (a.shape[0], a.shape[1]);
                    let p = b.shape[1];
                    assert_eq!(out.grad.len(), m * p, "Output grad has wrong size");

                    let kernel = select_kernel(m.max(n).max(p));
                    let d_c = out.grad.as_slice();

                    if a_req {
                        // B^T: (p x n)
                        let mut bt = pool.get_no_clear(p * n);
                        transpose_into(&b.data, n, p, &mut bt);

                        match kernel {
                            KernelType::Naive => matmul_naive_add_into_slices(d_c, m, p, &bt, n, &mut a.grad),
                            KernelType::Tiled => matmul_tiled_into_slices(d_c, m, p, &bt, n, &mut a.grad, 16),
                            KernelType::TiledMP => matmul_tiled_mp_into_slices(d_c, m, p, &bt, n, &mut a.grad, 16),
                        }

                        pool.release(bt);
                    }

                    if b_req {
                        // A^T: (n x m)
                        let mut at = pool.get_no_clear(n * m);
                        transpose_into(&a.data, m, n, &mut at);

                        match kernel {
                            KernelType::Naive => matmul_naive_add_into_slices(&at, n, m, d_c, p, &mut b.grad),
                            KernelType::Tiled => matmul_tiled_into_slices(&at, n, m, d_c, p, &mut b.grad, 16),
                            KernelType::TiledMP => matmul_tiled_mp_into_slices(&at, n, m, d_c, p, &mut b.grad, 16),
                        }

                        pool.release(at);
                    }
                }

                Operation::ReLU => {
                    let input_id = node.inputs[0];

                    if !store.get(input_id).requires_grad {
                        continue;
                    }

                    let (input, out) = store.get_mut_and_1(input_id, node.output);

                    for i in 0..out.grad.len() {
                        if input.data[i] > 0.0 {
                            input.grad[i] += out.grad[i];
                        }
                    }
                }

                Operation::MSE => {
                    // loss = mean((pred - target)^2)
                    // dloss/dpred = (2/N) * (pred - target) * upstream
                    // dloss/dtarget = -(2/N) * (pred - target) * upstream
                    let pred_id = node.inputs[0];
                    let target_id = node.inputs[1];

                    let pred_req = store.get(pred_id).requires_grad;
                    let target_req = store.get(target_id).requires_grad;

                    if pred_id == target_id {
                        continue;
                    }

                    let (pred, target, out) = store.get2_mut_and_1(pred_id, target_id, node.output);
                    let upstream = out.grad[0];

                    assert_eq!(
                        pred.data.len(),
                        target.data.len(),
                        "MSE backward: pred and target must have same length"
                    );

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
    use crate::tensor::tensor::{matmul, relu, Tensor};

    #[test]
    fn matmul_backward_matches_sums_when_loss_grad_is_ones() {
        let mut store = TensorStore::new();
        let mut graph = Graph::new();
        let mut pool = MemoryPool::new();
        pool.enabled = true;

        // A: 2x3, B: 3x2
        let a_id = store.add(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true));
        let b_id = store.add(Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], true));

        let out_id = matmul(a_id, b_id, &mut store, &mut graph);
        graph.backward(&mut store, out_id, &mut pool);

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

        graph.backward(&mut store, y_id, &mut pool);

        // dy/dx is 0 for x<=0, 1 for x>0 (with upstream grad = 1)
        assert_eq!(store.get(x_id).grad, vec![0.0, 1.0, 1.0, 0.0]);
    }

    
}