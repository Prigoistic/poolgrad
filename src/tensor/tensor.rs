use std::fmt;

use crate::autograd::graph::Graph;
use crate::autograd::node::{Node, Operation};

use crate::kernels::selector::{matmul_into, select_kernel_mm};
use crate::memory::pool::MemoryPool;
use crate::tensor::store::TensorStore;

pub struct Tensor {
    pub id: usize,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad: Vec<f32>,
    pub requires_grad: bool,
    pub creator: Option<usize>, // ID of the node that created this tensor
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(id={}, shape={:?}, requires_grad={}, creator={:?}, data={:?}",
            self.id, self.shape, self.requires_grad, self.creator, self.data
        )?;

        if self.requires_grad {
            write!(f, ", grad={:?}", self.grad)?;
        }

        write!(f, ")")
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let size: usize = shape.iter().product();

        assert_eq!(data.len(), size, "Shape does not match data length");

        // Grad buffers are allocated lazily during backward. This avoids allocating
        // for tensors that end up not being on the active gradient path.
        let grad = Vec::new();

        debug_assert!(requires_grad || grad.is_empty());

        Self {
            id: 0,
            data,
            grad,
            shape,
            requires_grad,
            creator: None,
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }
}

fn push_pointwise_binary_op(
    a_id: usize,
    b_id: usize,
    store: &mut TensorStore,
    graph: &mut Graph,
    _pool: Option<&mut MemoryPool>,
    op: Operation,
    f: impl Fn(f32, f32) -> f32,
) -> usize {
    let a = store.get(a_id);
    let b = store.get(b_id);

    assert_eq!(a.shape, b.shape, "pointwise op: shape mismatch");

    let result_data: Vec<f32> = a
        .data
        .iter()
        .copied()
        .zip(b.data.iter().copied())
        .map(|(x, y)| f(x, y))
        .collect();

    let requires_grad = a.requires_grad || b.requires_grad;
    let result = Tensor::new(result_data, a.shape.clone(), requires_grad);

    let out_id = store.add(result);
    let node_id = graph.add_node(Node {
        input0: a_id,
        input1: Some(b_id),
        output: out_id,
        op,
    });
    store.get_mut(out_id).creator = Some(node_id);
    out_id
}

#[cfg(test)]
pub fn add(a_id: usize, b_id: usize, store: &mut TensorStore, graph: &mut Graph) -> usize {
    push_pointwise_binary_op(a_id, b_id, store, graph, None, Operation::Add, |x, y| x + y)
}

pub fn add_with_pool(
    a_id: usize,
    b_id: usize,
    store: &mut TensorStore,
    graph: &mut Graph,
    pool: &mut MemoryPool,
) -> usize {
    push_pointwise_binary_op(
        a_id,
        b_id,
        store,
        graph,
        Some(pool),
        Operation::Add,
        |x, y| x + y,
    )
}

#[cfg(test)]
pub fn mul(a_id: usize, b_id: usize, store: &mut TensorStore, graph: &mut Graph) -> usize {
    push_pointwise_binary_op(a_id, b_id, store, graph, None, Operation::Mul, |x, y| x * y)
}

#[cfg(test)]
pub fn matmul(a_id: usize, b_id: usize, store: &mut TensorStore, graph: &mut Graph) -> usize {
    let a = store.get(a_id);
    let b = store.get(b_id);

    assert_eq!(a.shape.len(), 2, "First tensor must be 2D for matmul");
    assert_eq!(b.shape.len(), 2, "Second tensor must be 2D for matmul");
    assert_eq!(
        a.shape[1], b.shape[0],
        "Inner dimensions must match for matmul"
    );

    let (m, n) = (a.shape[0], a.shape[1]);
    let p = b.shape[1];

    let kernel = select_kernel_mm(m, n, p);
    let mut result_data = vec![0.0; m * p];
    matmul_into(a, b, kernel, &mut result_data);

    let requires_grad = a.requires_grad || b.requires_grad;
    let grad = if requires_grad {
        vec![0.0; m * p]
    } else {
        Vec::new()
    };

    let result_tensor = Tensor {
        id: 0, // will be set by TensorStore::add
        data: result_data,
        grad,
        shape: vec![m, p],
        requires_grad,
        creator: None,
    };

    let out_id = store.add(result_tensor);

    let node_id = graph.add_node(Node {
        input0: a_id,
        input1: Some(b_id),
        output: out_id,
        op: Operation::MatMul,
    });
    store.get_mut(out_id).creator = Some(node_id);

    out_id
}

pub fn matmul_with_pool(
    a_id: usize,
    b_id: usize,
    store: &mut TensorStore,
    graph: &mut Graph,
    _pool: &mut MemoryPool,
) -> usize {
    let a = store.get(a_id);
    let b = store.get(b_id);

    assert_eq!(a.shape.len(), 2, "First tensor must be 2D for matmul");
    assert_eq!(b.shape.len(), 2, "Second tensor must be 2D for matmul");
    assert_eq!(
        a.shape[1], b.shape[0],
        "Inner dimensions must match for matmul"
    );

    let (m, n) = (a.shape[0], a.shape[1]);
    let p = b.shape[1];

    let kernel = select_kernel_mm(m, n, p);
    let mut result_data = vec![0.0; m * p];
    matmul_into(a, b, kernel, &mut result_data);

    let requires_grad = a.requires_grad || b.requires_grad;
    let result_tensor = Tensor::new(result_data, vec![m, p], requires_grad);

    let out_id = store.add(result_tensor);

    let node_id = graph.add_node(Node {
        input0: a_id,
        input1: Some(b_id),
        output: out_id,
        op: Operation::MatMul,
    });
    store.get_mut(out_id).creator = Some(node_id);

    out_id
}

#[cfg(test)]
pub fn relu(input_id: usize, store: &mut TensorStore, graph: &mut Graph) -> usize {
    let input = store.get(input_id);

    let result_data: Vec<f32> = input
        .data
        .iter()
        .map(|x| if *x > 0.0 { *x } else { 0.0 })
        .collect();

    let result = Tensor::new(result_data, input.shape.clone(), input.requires_grad);

    let out_id = store.add(result);

    let node_id = graph.add_node(Node {
        input0: input_id,
        input1: None,
        output: out_id,
        op: Operation::ReLU,
    });
    store.get_mut(out_id).creator = Some(node_id);

    out_id
}
