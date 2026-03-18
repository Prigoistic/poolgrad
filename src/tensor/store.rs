use crate::tensor::tensor::Tensor;

pub struct TensorStore {
    pub tensors: Vec<Tensor>,
}

impl TensorStore {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    pub fn add(&mut self, tensor: Tensor) -> usize {
        let id = self.tensors.len();
        self.tensors.push(tensor);
        id
    }

    pub fn get(&self, id: usize) -> &Tensor {
        &self.tensors[id]
    }

    pub fn get_mut(&mut self, id: usize) -> &mut Tensor {
        &mut self.tensors[id]
    }
}

//Rust ownership rules will destroy you otherwise.
// This design:
// avoids borrowing issues
// centralizes tensors
// simplifies graph execution