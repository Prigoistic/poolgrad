#[derive(Clone)]

#[allow(dead_code)]
pub struct Node {
    pub inputs: Vec<usize>,
    pub output: usize,
    pub op: Operation,

}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum Operation {
    Add,
    Mul,
    MatMul,
}