use crate::tensor::tensor::Tensor;

pub struct TensorStore {
    tensors: Vec<Tensor>,
}

impl TensorStore {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    pub fn add(&mut self, mut tensor: Tensor) -> usize {
        let id = self.tensors.len();
        tensor.id = id;
        self.tensors.push(tensor);
        id
    }

    pub fn get(&self, id: usize) -> &Tensor {
        &self.tensors[id]
    }

    pub fn get_mut(&mut self, id: usize) -> &mut Tensor {
        &mut self.tensors[id]
    }

    pub fn get2_mut(&mut self, id1: usize, id2: usize) -> (&mut Tensor, &mut Tensor) {
        assert_ne!(id1, id2, "ids must be distinct; use get_mut for same-id case");

        let (low, high, low_first) = if id1 < id2 {
            (id1, id2, true)
        } else {
            (id2, id1, false)
        };

        let (left, right) = self.tensors.split_at_mut(high);
        let high_ref = &mut right[0];
        let low_ref = &mut left[low];

        if low_first {
            (low_ref, high_ref)
        } else {
            (high_ref, low_ref)
        }
    }
}

// Rust ownership rules will destroy you otherwise.