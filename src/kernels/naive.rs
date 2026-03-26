use crate::tensor::tensor::Tensor;

pub fn matmul_naive_into(a: &Tensor, b: &Tensor, out: &mut [f32]) {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);
    assert_eq!(out.len(), m * p);

    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a.data[i * n + k] * b.data[k * p + j];
            }
            out[i * p + j] = sum;
        }
    }
}

pub fn matmul_naive_add_into(a: &Tensor, b: &Tensor, out: &mut [f32]) {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);
    assert_eq!(out.len(), m * p);

    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a.data[i * n + k] * b.data[k * p + j];
            }
            out[i * p + j] += sum;
        }
    }
}

pub fn matmul_naive(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);

    let mut result = vec![0.0; m * p];
    matmul_naive_into(a, b, &mut result);

    Tensor::new(result, vec![m, p], a.requires_grad || b.requires_grad)
}