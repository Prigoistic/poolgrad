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

pub fn matmul_tiled_into_slices(
    a: &[f32],
    m: usize,
    n: usize,
    b: &[f32],
    p: usize,
    out: &mut [f32],
    block: usize,
) {
    assert_eq!(a.len(), m * n);
    assert_eq!(b.len(), n * p);
    assert_eq!(out.len(), m * p);

    if parallel_enabled() && m * p >= par_min_elems() && block > 0 {
        let rows_per_chunk = block.min(m).max(1);
        let chunk_elems = rows_per_chunk * p;

        out.par_chunks_mut(chunk_elems)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let ii = chunk_idx * rows_per_chunk;
                if ii >= m {
                    return;
                }

                let bm = ((ii + rows_per_chunk).min(m)) - ii;

                for jj in (0..p).step_by(block) {
                    for kk in (0..n).step_by(block) {
                        for i_local in 0..bm {
                            let i = ii + i_local;
                            for j in jj..(jj + block).min(p) {
                                let out_idx = i_local * p + j;
                                let mut sum = out_chunk[out_idx];
                                for k in kk..(kk + block).min(n) {
                                    sum += a[i * n + k] * b[k * p + j];
                                }
                                out_chunk[out_idx] = sum;
                            }
                        }
                    }
                }
            });
    } else {
        for ii in (0..m).step_by(block) {
            for jj in (0..p).step_by(block) {
                for kk in (0..n).step_by(block) {
                    for i in ii..(ii + block).min(m) {
                        for j in jj..(jj + block).min(p) {
                            let mut sum = out[i * p + j];
                            for k in kk..(kk + block).min(n) {
                                sum += a[i * n + k] * b[k * p + j];
                            }
                            out[i * p + j] = sum;
                        }
                    }
                }
            }
        }
    }
}

pub fn matmul_tiled_into(a: &Tensor, b: &Tensor, out: &mut [f32], block: usize) {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);
    matmul_tiled_into_slices(&a.data, m, n, &b.data, p, out, block);
}

pub fn matmul_tiled(a: &Tensor, b: &Tensor, block: usize) -> Tensor {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);

    let mut result = vec![0.0; m * p];
    matmul_tiled_into(a, b, &mut result, block);

    Tensor::new(result, vec![m, p], a.requires_grad || b.requires_grad)
}

/// Accumulates `out += a @ b^T` without materializing `b^T`.
///
/// Shapes:
/// - `a` is (m x k)
/// - `b` is (n x k) row-major
/// - `out` is (m x n)
pub fn matmul_tiled_add_into_slices_b_transposed(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
    block: usize,
) {
    assert!(block > 0);
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(out.len(), m * n);

    if parallel_enabled() && m * n >= par_min_elems() {
        out.par_chunks_mut(n).enumerate().for_each(|(i, out_row)| {
            let a_row = &a[i * k..(i + 1) * k];
            for kk0 in (0..k).step_by(block) {
                let kk1 = (kk0 + block).min(k);
                for j in 0..n {
                    let b_row = &b[j * k..(j + 1) * k];
                    let mut sum = 0.0;
                    for kk in kk0..kk1 {
                        sum += a_row[kk] * b_row[kk];
                    }
                    out_row[j] += sum;
                }
            }
        });
    } else {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            for kk0 in (0..k).step_by(block) {
                let kk1 = (kk0 + block).min(k);
                for j in 0..n {
                    let b_row = &b[j * k..(j + 1) * k];
                    let mut sum = 0.0;
                    for kk in kk0..kk1 {
                        sum += a_row[kk] * b_row[kk];
                    }
                    out[i * n + j] += sum;
                }
            }
        }
    }
}

/// Accumulates `out += a^T @ b` without materializing `a^T`.
///
/// Shapes:
/// - `a` is (m x n) row-major
/// - `b` is (m x p) row-major
/// - `out` is (n x p)
pub fn matmul_tiled_add_into_slices_a_transposed(
    a: &[f32],
    m: usize,
    n: usize,
    b: &[f32],
    p: usize,
    out: &mut [f32],
    block: usize,
) {
    assert!(block > 0);
    assert_eq!(a.len(), m * n);
    assert_eq!(b.len(), m * p);
    assert_eq!(out.len(), n * p);

    if parallel_enabled() && n * p >= par_min_elems() {
        out.par_chunks_mut(p).enumerate().for_each(|(i, out_row)| {
            for kk0 in (0..m).step_by(block) {
                let kk1 = (kk0 + block).min(m);
                for j in 0..p {
                    let mut sum = 0.0;
                    for kk in kk0..kk1 {
                        sum += a[kk * n + i] * b[kk * p + j];
                    }
                    out_row[j] += sum;
                }
            }
        });
    } else {
        for i in 0..n {
            for kk0 in (0..m).step_by(block) {
                let kk1 = (kk0 + block).min(m);
                for j in 0..p {
                    let mut sum = 0.0;
                    for kk in kk0..kk1 {
                        sum += a[kk * n + i] * b[kk * p + j];
                    }
                    out[i * p + j] += sum;
                }
            }
        }
    }
}