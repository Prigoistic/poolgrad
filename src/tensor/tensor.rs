#![allow(dead_code)]

use std::fmt;

use crate::autograd::graph::Graph;
use crate::autograd::node::{Node, Operation};

use crate::tensor::store::TensorStore;
use crate::mem::pool::MemoryPool;
use crate::kernels::naive::matmul_naive_into;
use crate::kernels::tiled::matmul_tiled_into;
use crate::kernels::selector::{select_kernel, KernelType};

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

        let grad = if requires_grad {
            vec![0.0; size]
        } else {
            Vec::new()
        };

        Self {
            id: 0,
            data,
            grad,
            shape,
            requires_grad,
            creator: None,
        }
    }

    pub fn new_with_pool(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
        pool: &mut MemoryPool,
    ) -> Self {
        let size: usize = shape.iter().product();

        assert_eq!(data.len(), size, "Shape does not match data length");

        let grad = if requires_grad { pool.get(size) } else { Vec::new() };

        Self {
            id: 0,
            data,
            grad,
            shape,
            requires_grad,
            creator: None,
        }
    }

    pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> Self {
        let size: usize = shape.iter().product();
        Self::new(vec![0.0; size], shape, requires_grad)
    }

    pub fn ones(shape: Vec<usize>, requires_grad: bool) -> Self {
        let size: usize = shape.iter().product();
        Self::new(vec![1.0; size], shape, requires_grad)
    }

    pub fn zero_grad(&mut self) {
        for g in self.grad.iter_mut() {
            *g = 0.0;
        }
    }
//implemented add operations for tensors 
    pub fn add(
    a_id: usize,
    b_id: usize,
    store: &mut TensorStore,
    graph: &mut Graph,
) -> usize {
    let a = store.get(a_id);
    let b = store.get(b_id);

    assert_eq!(a.shape, b.shape);

    let result_data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .collect();

    let requires_grad = a.requires_grad || b.requires_grad;

    let grad = if requires_grad {
        vec![0.0; a.data.len()]
    } else {
        Vec::new()
    };

    let result_tensor = Tensor {
        id: 0, // temporary
        data: result_data,
        grad,
        shape: a.shape.clone(),
        requires_grad,
        creator: None,
    };

    let out_id = store.add(result_tensor);

    let node = Node {
        inputs: vec![a_id, b_id],
        output: out_id,
        op: Operation::Add,
    };

    let node_id = graph.nodes.len();
    graph.add_node(node);

    store.get_mut(out_id).creator = Some(node_id);

    out_id
}

    pub fn add_with_pool(
        a_id: usize,
        b_id: usize,
        store: &mut TensorStore,
        graph: &mut Graph,
        pool: &mut MemoryPool,
    ) -> usize {
        let a = store.get(a_id);
        let b = store.get(b_id);

        assert_eq!(a.shape, b.shape);

        let result_data: Vec<f32> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| x + y)
            .collect();

        let requires_grad = a.requires_grad || b.requires_grad;
        let grad = if requires_grad {
            pool.get(a.data.len())
        } else {
            vec![0.0; a.data.len()]
        };

        let result_tensor = Tensor {
            id: 0, // temporary
            data: result_data,
            grad,
            shape: a.shape.clone(),
            requires_grad,
            creator: None,
        };

        let out_id = store.add(result_tensor);

        let node = Node {
            inputs: vec![a_id, b_id],
            output: out_id,
            op: Operation::Add,
        };

        let node_id = graph.nodes.len();
        graph.add_node(node);

        store.get_mut(out_id).creator = Some(node_id);

        out_id
    }

    pub fn mul(&self,other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shapes must be the same for multiplication");

        let result_data: Vec<f32> = self.data.iter().zip(other.data.iter()).map(|(a,b)| a*b).collect();

        Tensor::new(
            result_data,
            self.shape.clone(),
            self.requires_grad || other.requires_grad

        )

    }   

    pub fn matmul(&self,other: &Tensor) -> Tensor{
        assert_eq!(self.shape.len(), 2, "First tensor must be 2D for matmul");
        assert_eq!(other.shape.len(), 2, "Second tensor must be 2D for matmul");
        assert_eq!(self.shape[1], other.shape[0], "Inner dimensions must match for matmul");

       let (m,n) = (self.shape[0], self.shape[1]);
       let (_n2,p) = (other.shape[0], other.shape[1]);   


        let mut result_data = vec![0.0; m*p];

        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    let a = self.data[i*n + k];
                    let b = other.data[k*p + j];
                    sum += a * b;
                }
                result_data[i*p + j] = sum;
            }
        }

        Tensor::new(
            result_data,
            vec![m,p],
            self.requires_grad || other.requires_grad
        )
    }
}   

pub fn mul(a_id: usize, b_id: usize, store: &mut TensorStore, graph: &mut Graph) -> usize {
    let a = store.get(a_id);
    let b = store.get(b_id);

    assert_eq!(a.shape, b.shape, "Shapes must be the same for multiplication");

    let result_data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x * y)
        .collect();

    let requires_grad = a.requires_grad || b.requires_grad;

    let grad = if requires_grad {
        vec![0.0; a.data.len()]
    } else {
        Vec::new()
    };

    let result_tensor = Tensor {
        id: 0, // will be set by TensorStore::add
        data: result_data,
        grad,
        shape: a.shape.clone(),
        requires_grad,
        creator: None,
    };

    let out_id = store.add(result_tensor);

    let node = Node {
        inputs: vec![a_id, b_id],
        output: out_id,
        op: Operation::Mul,
    };

    let node_id = graph.nodes.len();
    graph.add_node(node);

    store.get_mut(out_id).creator = Some(node_id);

    out_id
}

pub fn matmul(a_id: usize, b_id: usize, store: &mut TensorStore, graph: &mut Graph) -> usize {
    let a = store.get(a_id);
    let b = store.get(b_id);

    assert_eq!(a.shape.len(), 2, "First tensor must be 2D for matmul");
    assert_eq!(b.shape.len(), 2, "Second tensor must be 2D for matmul");
    assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match for matmul");

    let (m, n) = (a.shape[0], a.shape[1]);
    let p = b.shape[1];

    let mut result_data = vec![0.0; m * p];
    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a.data[i * n + k] * b.data[k * p + j];
            }
            result_data[i * p + j] = sum;
        }
    }

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

    let node = Node {
        inputs: vec![a_id, b_id],
        output: out_id,
        op: Operation::MatMul,
    };

    let node_id = graph.nodes.len();
    graph.add_node(node);
    store.get_mut(out_id).creator = Some(node_id);

    out_id
}

pub fn matmul_scheduled_with_pool(
    a_id: usize,
    b_id: usize,
    store: &mut TensorStore,
    graph: &mut Graph,
    pool: &mut MemoryPool,
) -> usize {
    let a = store.get(a_id);
    let b = store.get(b_id);

    assert_eq!(a.shape.len(), 2, "First tensor must be 2D for matmul");
    assert_eq!(b.shape.len(), 2, "Second tensor must be 2D for matmul");
    assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match for matmul");

    let (m, n) = (a.shape[0], a.shape[1]);
    let p = b.shape[1];

    let size_hint = m.max(n).max(p);
    let kernel = select_kernel(size_hint);

    let mut result_data = vec![0.0; m * p];
    match kernel {
        KernelType::Naive => matmul_naive_into(a, b, &mut result_data),
        KernelType::Tiled => matmul_tiled_into(a, b, &mut result_data, 16),
        KernelType::TiledMP => crate::kernels::tiled_mp::matmul_tiled_mp_into(a, b, &mut result_data, 16),
    }

    let requires_grad = a.requires_grad || b.requires_grad;
    let grad = if requires_grad { pool.get(m * p) } else { Vec::new() };

    let result_tensor = Tensor {
        id: 0, // will be set by TensorStore::add
        data: result_data,
        grad,
        shape: vec![m, p],
        requires_grad,
        creator: None,
    };

    let out_id = store.add(result_tensor);

    let node = Node {
        inputs: vec![a_id, b_id],
        output: out_id,
        op: Operation::MatMul,
    };

    let node_id = graph.nodes.len();
    graph.add_node(node);
    store.get_mut(out_id).creator = Some(node_id);

    out_id
}

pub fn relu(input_id: usize, store: &mut TensorStore, graph: &mut Graph) -> usize {
    let input = store.get(input_id);

    let result_data: Vec<f32> = input
        .data
        .iter()
        .map(|x| if *x > 0.0 { *x } else { 0.0 })
        .collect();

    let result = Tensor::new(result_data, input.shape.clone(), input.requires_grad);

    let out_id = store.add(result);

    let node = Node {
        inputs: vec![input_id],
        output: out_id,
        op: Operation::ReLU,
    };

    let node_id = graph.nodes.len();
    graph.add_node(node);

    store.get_mut(out_id).creator = Some(node_id);

    out_id
}
