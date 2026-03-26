use crate::tensor::tensor::Tensor;

pub fn matmul_tiled_into(a: &Tensor, b: &Tensor, out: &mut [f32], block: usize) {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);
    assert_eq!(out.len(), m * p);

    for ii in (0..m).step_by(block) {
        for jj in (0..p).step_by(block) {
            for kk in (0..n).step_by(block) {
                for i in ii..(ii + block).min(m) {
                    for j in jj..(jj + block).min(p) {
                        let mut sum = out[i * p + j];
                        for k in kk..(kk + block).min(n) {
                            sum += a.data[i * n + k] * b.data[k * p + j];
                        }
                        out[i * p + j] = sum;
                    }
                }
            }
        }
    }
}

pub fn matmul_tiled(a: &Tensor, b: &Tensor, block: usize) -> Tensor {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);

    let mut result = vec![0.0; m * p];
    matmul_tiled_into(a, b, &mut result, block);

    Tensor::new(result, vec![m, p], a.requires_grad || b.requires_grad)
}