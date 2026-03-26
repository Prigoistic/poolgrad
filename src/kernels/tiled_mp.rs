#![allow(dead_code)]

use crate::kernels::mp::{mp_block_mul_add, MPTransform, MPScratch};
use crate::kernels::tiled::matmul_tiled_into;
use crate::tensor::tensor::Tensor;

fn tiled_style_block_mul_accum(
    a: &[f32],
    a_ld: usize,
    a_row: usize,
    a_col: usize,
    b: &[f32],
    b_ld: usize,
    b_row: usize,
    b_col: usize,
    c: &mut [f32],
    c_ld: usize,
    c_row: usize,
    c_col: usize,
    m: usize,
    n: usize,
    p: usize,
) {
    for i in 0..m {
        for j in 0..p {
            let out_idx = (c_row + i) * c_ld + (c_col + j);
            let mut sum = c[out_idx];
            for k in 0..n {
                sum += a[(a_row + i) * a_ld + (a_col + k)]
                    * b[(b_row + k) * b_ld + (b_col + j)];
            }
            c[out_idx] = sum;
        }
    }
}

pub fn matmul_tiled_mp_into(a: &Tensor, b: &Tensor, out: &mut [f32], block: usize) {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);
    assert_eq!(out.len(), m * p);

    let size_hint = m.max(n).max(p);
    let mp_max_size = std::env::var("POOLGRAD_MP_MAX_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(128);
    let mp_enabled = size_hint <= mp_max_size;

    if !mp_enabled {
        // Exact fallback: reuse the existing tiled kernel, which matches naive accumulation order.
        matmul_tiled_into(a, b, out, block);
        return;
    }

    let transform = MPTransform::strassen();
    let mut scratch = MPScratch::new();

    for ii in (0..m).step_by(block) {
        for jj in (0..p).step_by(block) {
            for kk in (0..n).step_by(block) {
                let bm = (ii + block).min(m) - ii;
                let bp = (jj + block).min(p) - jj;
                let bn = (kk + block).min(n) - kk;

                // MP only supports full, square blocks here, and is guarded for numerical stability.
                if mp_enabled && bm == block && bp == block && bn == block {
                    let applied = mp_block_mul_add(
                        &a.data,
                        n,
                        ii,
                        kk,
                        &b.data,
                        p,
                        kk,
                        jj,
                        out,
                        p,
                        ii,
                        jj,
                        block,
                        &transform,
                        &mut scratch,
                    );
                    if applied {
                        continue;
                    }
                }

                // Fallback for edges or unsupported block sizes.
                tiled_style_block_mul_accum(
                    &a.data, n, ii, kk,
                    &b.data, p, kk, jj,
                    out, p, ii, jj,
                    bm, bn, bp,
                );
            }
        }
    }
}

pub fn matmul_tiled_mp(a: &Tensor, b: &Tensor, block: usize) -> Tensor {
    let (m, _n) = (a.shape[0], a.shape[1]);
    let p = b.shape[1];

    let mut result = vec![0.0; m * p];
    matmul_tiled_mp_into(a, b, &mut result, block);

    Tensor::new(result, vec![m, p], a.requires_grad || b.requires_grad)
}
