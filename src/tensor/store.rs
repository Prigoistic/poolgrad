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

    #[allow(dead_code)]
    pub fn get2_mut(&mut self, id1: usize, id2: usize) -> (&mut Tensor, &mut Tensor) {
        assert_ne!(
            id1, id2,
            "ids must be distinct; use get_mut for same-id case"
        );

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

    pub fn get_mut_and_1(&mut self, id_mut: usize, id_immut: usize) -> (&mut Tensor, &Tensor) {
        assert_ne!(id_mut, id_immut, "ids must be distinct");

        let (low, high, mut_is_low) = if id_mut < id_immut {
            (id_mut, id_immut, true)
        } else {
            (id_immut, id_mut, false)
        };

        let (left, right) = self.tensors.split_at_mut(high);
        let high_ref = &mut right[0];
        let low_ref = &mut left[low];

        if mut_is_low {
            (low_ref, &*high_ref)
        } else {
            (high_ref, &*low_ref)
        }
    }

    pub fn get2_mut_and_1(
        &mut self,
        id_mut1: usize,
        id_mut2: usize,
        id_immut: usize,
    ) -> (&mut Tensor, &mut Tensor, &Tensor) {
        assert_ne!(id_mut1, id_mut2, "mutable ids must be distinct");
        assert_ne!(id_mut1, id_immut, "ids must be distinct");
        assert_ne!(id_mut2, id_immut, "ids must be distinct");

        let mut ids = [(id_mut1, 0usize), (id_mut2, 1usize), (id_immut, 2usize)];
        ids.sort_by_key(|(id, _role)| *id);

        let (id0, role0) = ids[0];
        let (id1, role1) = ids[1];
        let (id2, role2) = ids[2];

        let (s01, s2) = self.tensors.split_at_mut(id2);
        let t2 = &mut s2[0];
        let (s0, s1) = s01.split_at_mut(id1);
        let t1 = &mut s1[0];
        let t0 = &mut s0[id0];

        match (role0, role1, role2) {
            (0, 1, 2) => (t0, t1, &*t2),
            (1, 0, 2) => (t1, t0, &*t2),
            (0, 2, 1) => (t0, t2, &*t1),
            (2, 0, 1) => (t2, t0, &*t1),
            (1, 2, 0) => (t2, t1, &*t0),
            (2, 1, 0) => (t1, t2, &*t0),
            _ => unreachable!("invalid role permutation"),
        }
    }
}

// Rust ownership rules will destroy you otherwise.
