#[derive(Clone)]

#[allow(dead_code)]
pub struct Node {
    pub input0: usize,
    pub input1: Option<usize>,
    pub output: usize,
    pub op: Operation,

}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum Operation {
    Add,
    Mul,
    MatMul,
    ReLU,
    MSE,
}