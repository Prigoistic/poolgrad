#[derive(Clone)]
pub struct Node {
    pub input0: usize,
    pub input1: Option<usize>,
    pub output: usize,
    pub op: Operation,
}

#[derive(Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub enum Operation {
    Add,
    #[allow(dead_code)]
    Mul,
    MatMul,
    #[allow(dead_code)]
    ReLU,
    MSE,
}
