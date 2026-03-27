use crate::tensor::tensor::Tensor;
use rayon::prelude::*;

fn parallel_enabled() -> bool {
    std::env::var("POOLGRAD_PAR")
        .ok()
        .as_deref()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
}

fn par_min_elems() -> usize {
    std::env::var("POOLGRAD_PAR_MIN_ELEMS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16 * 1024)
}

pub fn matmul_naive_into(a: &Tensor, b: &Tensor, out: &mut [f32]) {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);
    assert_eq!(out.len(), m * p);

    if parallel_enabled() && m * p >= par_min_elems() {
        out.par_chunks_mut(p).enumerate().for_each(|(i, out_row)| {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a.data[i * n + k] * b.data[k * p + j];
                }
                out_row[j] = sum;
            }
        });
    } else {
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
}

pub fn matmul_naive_add_into(a: &Tensor, b: &Tensor, out: &mut [f32]) {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);
    assert_eq!(out.len(), m * p);

    if parallel_enabled() && m * p >= par_min_elems() {
        out.par_chunks_mut(p).enumerate().for_each(|(i, out_row)| {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a.data[i * n + k] * b.data[k * p + j];
                }
                out_row[j] += sum;
            }
        });
    } else {
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
}

pub fn matmul_naive(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);

    let mut result = vec![0.0; m * p];
    matmul_naive_into(a, b, &mut result);

    Tensor::new(result, vec![m, p], a.requires_grad || b.requires_grad)
}