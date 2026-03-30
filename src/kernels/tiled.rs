use crate::config::{par_min_elems, parallel_enabled};
use crate::tensor::tensor::Tensor;
use rayon::prelude::*;
use std::sync::OnceLock;

const MR: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MicroKernel {
    Scalar4x4,
    Neon4x4,
    #[allow(dead_code)]
    Avx2Fma4x8,
}

impl MicroKernel {
    fn nr(self) -> usize {
        match self {
            MicroKernel::Scalar4x4 => 4,
            MicroKernel::Neon4x4 => 4,
            MicroKernel::Avx2Fma4x8 => 8,
        }
    }
}

fn pick_microkernel() -> MicroKernel {
    static MK: OnceLock<MicroKernel> = OnceLock::new();
    *MK.get_or_init(|| {
        if let Ok(v) = std::env::var("POOLGRAD_MICROKERNEL") {
            match v.trim().to_ascii_lowercase().as_str() {
                "scalar" | "scalar4x4" => return MicroKernel::Scalar4x4,
                "neon" | "neon4x4" => {
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") {
                            return MicroKernel::Neon4x4;
                        }
                    }
                    return MicroKernel::Scalar4x4;
                }
                "avx2" | "avx2fma" | "avx2_fma" | "avx2fma4x8" => {
                    #[cfg(target_arch = "x86_64")]
                    {
                        if std::arch::is_x86_feature_detected!("avx2")
                            && std::arch::is_x86_feature_detected!("fma")
                        {
                            return MicroKernel::Avx2Fma4x8;
                        }
                    }
                    return MicroKernel::Scalar4x4;
                }
                _ => {}
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is effectively baseline on modern AArch64, but keep the feature check.
            if std::arch::is_aarch64_feature_detected!("neon") {
                return MicroKernel::Neon4x4;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return MicroKernel::Avx2Fma4x8;
            }
        }

        MicroKernel::Scalar4x4
    })
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn pack_b_panel_nn(
    b: &[f32],
    _n: usize,
    p: usize,
    k0: usize,
    j0: usize,
    kc: usize,
    nr: usize,
    packed: &mut [f32],
) {
    debug_assert_eq!(packed.len(), kc * nr);
    for kk in 0..kc {
        let k = k0 + kk;
        let src_row = &b[k * p..(k + 1) * p];
        let dst = &mut packed[kk * nr..(kk + 1) * nr];
        for (j, dst_j) in dst.iter_mut().enumerate() {
            let col = j0 + j;
            *dst_j = if col < p { src_row[col] } else { 0.0 };
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn pack_a_panel_tn(
    a: &[f32],
    m: usize,
    n: usize,
    i0: usize,
    k0: usize,
    mr: usize,
    kc: usize,
    packed: &mut [f32],
) {
    // Pack A^T panel (mr x kc) where A is (m x n), so A^T is (n x m).
    // packed[r*kc + kk] = A[(k0+kk), (i0+r)]
    debug_assert_eq!(packed.len(), mr * kc);
    for r in 0..mr {
        let col = i0 + r;
        for kk in 0..kc {
            let row = k0 + kk;
            packed[r * kc + kk] = a[row * n + col];
        }
    }
    // Silence unused warnings when compiled for generic targets.
    let _ = m;
}

#[inline]
fn microkernel_scalar_4x4(
    a: &[f32],
    a_rs: usize,
    packed_b: &[f32],
    kc: usize,
    c: &mut [f32],
    c_rs: usize,
) {
    let mut acc0 = [0.0f32; 4];
    let mut acc1 = [0.0f32; 4];
    let mut acc2 = [0.0f32; 4];
    let mut acc3 = [0.0f32; 4];

    for j in 0..4 {
        acc0[j] = c[j];
        acc1[j] = c[c_rs + j];
        acc2[j] = c[2 * c_rs + j];
        acc3[j] = c[3 * c_rs + j];
    }

    for kk in 0..kc {
        let b0 = packed_b[kk * 4];
        let b1 = packed_b[kk * 4 + 1];
        let b2 = packed_b[kk * 4 + 2];
        let b3 = packed_b[kk * 4 + 3];

        let a0 = a[kk];
        let a1 = a[a_rs + kk];
        let a2 = a[2 * a_rs + kk];
        let a3 = a[3 * a_rs + kk];

        acc0[0] += a0 * b0;
        acc0[1] += a0 * b1;
        acc0[2] += a0 * b2;
        acc0[3] += a0 * b3;

        acc1[0] += a1 * b0;
        acc1[1] += a1 * b1;
        acc1[2] += a1 * b2;
        acc1[3] += a1 * b3;

        acc2[0] += a2 * b0;
        acc2[1] += a2 * b1;
        acc2[2] += a2 * b2;
        acc2[3] += a2 * b3;

        acc3[0] += a3 * b0;
        acc3[1] += a3 * b1;
        acc3[2] += a3 * b2;
        acc3[3] += a3 * b3;
    }

    for j in 0..4 {
        c[j] = acc0[j];
        c[c_rs + j] = acc1[j];
        c[2 * c_rs + j] = acc2[j];
        c[3 * c_rs + j] = acc3[j];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn microkernel_neon_4x4(
    a: &[f32],
    a_rs: usize,
    packed_b: &[f32],
    kc: usize,
    c: &mut [f32],
    c_rs: usize,
) {
    use core::arch::aarch64::*;

    unsafe {
        let mut acc0 = vld1q_f32(c.as_ptr());
        let mut acc1 = vld1q_f32(c.as_ptr().add(c_rs));
        let mut acc2 = vld1q_f32(c.as_ptr().add(2 * c_rs));
        let mut acc3 = vld1q_f32(c.as_ptr().add(3 * c_rs));

        for kk in 0..kc {
            let b = vld1q_f32(packed_b.as_ptr().add(kk * 4));
            acc0 = vmlaq_n_f32(acc0, b, *a.get_unchecked(kk));
            acc1 = vmlaq_n_f32(acc1, b, *a.get_unchecked(a_rs + kk));
            acc2 = vmlaq_n_f32(acc2, b, *a.get_unchecked(2 * a_rs + kk));
            acc3 = vmlaq_n_f32(acc3, b, *a.get_unchecked(3 * a_rs + kk));
        }

        vst1q_f32(c.as_mut_ptr(), acc0);
        vst1q_f32(c.as_mut_ptr().add(c_rs), acc1);
        vst1q_f32(c.as_mut_ptr().add(2 * c_rs), acc2);
        vst1q_f32(c.as_mut_ptr().add(3 * c_rs), acc3);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_avx2_fma_4x8(
    a: &[f32],
    a_rs: usize,
    packed_b: &[f32],
    kc: usize,
    c: &mut [f32],
    c_rs: usize,
) {
    use core::arch::x86_64::*;

    unsafe {
        let mut acc0 = _mm256_loadu_ps(c.as_ptr().add(0 * c_rs));
        let mut acc1 = _mm256_loadu_ps(c.as_ptr().add(1 * c_rs));
        let mut acc2 = _mm256_loadu_ps(c.as_ptr().add(2 * c_rs));
        let mut acc3 = _mm256_loadu_ps(c.as_ptr().add(3 * c_rs));

        for kk in 0..kc {
            let b = _mm256_loadu_ps(packed_b.as_ptr().add(kk * 8));
            let a0 = _mm256_set1_ps(*a.get_unchecked(0 * a_rs + kk));
            let a1 = _mm256_set1_ps(*a.get_unchecked(1 * a_rs + kk));
            let a2 = _mm256_set1_ps(*a.get_unchecked(2 * a_rs + kk));
            let a3 = _mm256_set1_ps(*a.get_unchecked(3 * a_rs + kk));
            acc0 = _mm256_fmadd_ps(a0, b, acc0);
            acc1 = _mm256_fmadd_ps(a1, b, acc1);
            acc2 = _mm256_fmadd_ps(a2, b, acc2);
            acc3 = _mm256_fmadd_ps(a3, b, acc3);
        }

        _mm256_storeu_ps(c.as_mut_ptr().add(0 * c_rs), acc0);
        _mm256_storeu_ps(c.as_mut_ptr().add(1 * c_rs), acc1);
        _mm256_storeu_ps(c.as_mut_ptr().add(2 * c_rs), acc2);
        _mm256_storeu_ps(c.as_mut_ptr().add(3 * c_rs), acc3);
    }
}

fn matmul_tiled_nn_packed_add_into(
    a: &[f32],
    m: usize,
    n: usize,
    b: &[f32],
    p: usize,
    out: &mut [f32],
    block: usize,
) {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), n * p);
    debug_assert_eq!(out.len(), m * p);

    let mk = pick_microkernel();
    let nr = mk.nr();
    let kc_max = block.max(1);

    // Pack-and-compute per NR-wide column panel. This lets us share a single packed-B
    // panel across all row chunks, avoiding repacking `b` per slab.
    // Packed kernels tend to have higher per-task overhead (packing, microkernel setup),
    // so use a higher threshold before turning on Rayon.
    let use_par = parallel_enabled() && m * p >= par_min_elems().saturating_mul(8);
    // Use coarser chunks than `block` to keep Rayon overhead amortized.
    let rows_per_chunk = block.saturating_mul(4).max(MR).min(m).max(1);
    let chunk_elems = rows_per_chunk * p;

    let mut packed_b_full = vec![0.0f32; n * nr];

    for jj in (0..p).step_by(nr) {
        let nr_eff = (p - jj).min(nr);
        if nr_eff != nr {
            // Remainder columns: keep a simple scalar path.
            for kk0 in (0..n).step_by(kc_max) {
                let kc = (n - kk0).min(kc_max);
                for ii in 0..m {
                    for j in jj..(jj + nr_eff) {
                        let mut sum = out[ii * p + j];
                        for t in 0..kc {
                            sum += a[ii * n + (kk0 + t)] * b[(kk0 + t) * p + j];
                        }
                        out[ii * p + j] = sum;
                    }
                }
            }
            continue;
        }

        // Full panel: pack B(jj..jj+nr) for all k.
        if packed_b_full.len() != n * nr {
            packed_b_full.resize(n * nr, 0.0);
        }
        for kk0 in 0..n {
            let src = &b[kk0 * p + jj..kk0 * p + (jj + nr)];
            let dst = &mut packed_b_full[kk0 * nr..(kk0 + 1) * nr];
            dst.copy_from_slice(src);
        }
        let packed_panel = &packed_b_full[..n * nr];

        if use_par {
            out.par_chunks_mut(chunk_elems)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let ii0 = chunk_idx * rows_per_chunk;
                    if ii0 >= m {
                        return;
                    }
                    let bm = ((ii0 + rows_per_chunk).min(m)) - ii0;

                    for kk0 in (0..n).step_by(kc_max) {
                        let kc = (n - kk0).min(kc_max);
                        let packed_b = &packed_panel[kk0 * nr..(kk0 + kc) * nr];

                        for i_local in (0..bm).step_by(MR) {
                            let mr = (bm - i_local).min(MR);
                            if mr == MR {
                                let a_ptr = &a[(ii0 + i_local) * n + kk0..];
                                let c_ptr = &mut out_chunk[i_local * p + jj..];
                                match mk {
                                    MicroKernel::Scalar4x4 if nr == 4 => {
                                        microkernel_scalar_4x4(a_ptr, n, packed_b, kc, c_ptr, p)
                                    }
                                    MicroKernel::Neon4x4 if nr == 4 => {
                                        #[cfg(target_arch = "aarch64")]
                                        unsafe {
                                            microkernel_neon_4x4(a_ptr, n, packed_b, kc, c_ptr, p)
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        microkernel_scalar_4x4(a_ptr, n, packed_b, kc, c_ptr, p)
                                    }
                                    MicroKernel::Avx2Fma4x8 if nr == 8 => {
                                        #[cfg(target_arch = "x86_64")]
                                        unsafe {
                                            microkernel_avx2_fma_4x8(
                                                a_ptr, n, packed_b, kc, c_ptr, p,
                                            )
                                        }
                                        #[cfg(not(target_arch = "x86_64"))]
                                        {
                                            for i in 0..MR {
                                                for j in 0..nr {
                                                    let mut sum = c_ptr[i * p + j];
                                                    for t in 0..kc {
                                                        sum +=
                                                            a_ptr[i * n + t] * packed_b[t * nr + j];
                                                    }
                                                    c_ptr[i * p + j] = sum;
                                                }
                                            }
                                        }
                                    }
                                    _ => {
                                        for i in 0..MR {
                                            for j in 0..nr {
                                                let mut sum = c_ptr[i * p + j];
                                                for t in 0..kc {
                                                    sum += a_ptr[i * n + t] * packed_b[t * nr + j];
                                                }
                                                c_ptr[i * p + j] = sum;
                                            }
                                        }
                                    }
                                }
                            } else {
                                for i in 0..mr {
                                    for j in 0..nr {
                                        let mut sum = out_chunk[(i_local + i) * p + (jj + j)];
                                        for t in 0..kc {
                                            sum += a[(ii0 + i_local + i) * n + (kk0 + t)]
                                                * b[(kk0 + t) * p + (jj + j)];
                                        }
                                        out_chunk[(i_local + i) * p + (jj + j)] = sum;
                                    }
                                }
                            }
                        }
                    }
                });
        } else {
            for kk0 in (0..n).step_by(kc_max) {
                let kc = (n - kk0).min(kc_max);
                let packed_b = &packed_panel[kk0 * nr..(kk0 + kc) * nr];

                for ii in (0..m).step_by(MR) {
                    let mr = (m - ii).min(MR);
                    if mr == MR {
                        let a_ptr = &a[ii * n + kk0..];
                        let c_ptr = &mut out[ii * p + jj..];
                        match mk {
                            MicroKernel::Scalar4x4 if nr == 4 => {
                                microkernel_scalar_4x4(a_ptr, n, packed_b, kc, c_ptr, p)
                            }
                            MicroKernel::Neon4x4 if nr == 4 => {
                                #[cfg(target_arch = "aarch64")]
                                unsafe {
                                    microkernel_neon_4x4(a_ptr, n, packed_b, kc, c_ptr, p)
                                }
                                #[cfg(not(target_arch = "aarch64"))]
                                microkernel_scalar_4x4(a_ptr, n, packed_b, kc, c_ptr, p)
                            }
                            MicroKernel::Avx2Fma4x8 if nr == 8 => {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    microkernel_avx2_fma_4x8(a_ptr, n, packed_b, kc, c_ptr, p)
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    for i in 0..MR {
                                        for j in 0..nr {
                                            let mut sum = c_ptr[i * p + j];
                                            for t in 0..kc {
                                                sum += a_ptr[i * n + t] * packed_b[t * nr + j];
                                            }
                                            c_ptr[i * p + j] = sum;
                                        }
                                    }
                                }
                            }
                            _ => {
                                for i in 0..MR {
                                    for j in 0..nr {
                                        let mut sum = c_ptr[i * p + j];
                                        for t in 0..kc {
                                            sum += a_ptr[i * n + t] * packed_b[t * nr + j];
                                        }
                                        c_ptr[i * p + j] = sum;
                                    }
                                }
                            }
                        }
                    } else {
                        for i in 0..mr {
                            for j in 0..nr {
                                let mut sum = out[(ii + i) * p + (jj + j)];
                                for t in 0..kc {
                                    sum +=
                                        a[(ii + i) * n + (kk0 + t)] * b[(kk0 + t) * p + (jj + j)];
                                }
                                out[(ii + i) * p + (jj + j)] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn matmul_tiled_nt_packed_add_into(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
    block: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(out.len(), m * n);

    let mk = pick_microkernel();
    let nr = mk.nr();
    let kc_max = block.max(1);

    // Packed kernels tend to have higher per-task overhead (packing, microkernel setup),
    // so use a higher threshold before turning on Rayon.
    let use_par = parallel_enabled() && m * n >= par_min_elems().saturating_mul(8);
    // Use coarser chunks than `block` to keep Rayon overhead amortized.
    let rows_per_chunk = block.saturating_mul(4).max(MR).min(m).max(1);
    let chunk_elems = rows_per_chunk * n;

    let mut packed_b_full = vec![0.0f32; k * nr];

    for jj in (0..n).step_by(nr) {
        let nr_eff = (n - jj).min(nr);
        if nr_eff != nr {
            // Remainder columns: scalar fallback.
            for kk0 in (0..k).step_by(kc_max) {
                let kc = (k - kk0).min(kc_max);
                for ii in 0..m {
                    for j in jj..(jj + nr_eff) {
                        let mut sum = out[ii * n + j];
                        for t in 0..kc {
                            sum += a[ii * k + (kk0 + t)] * b[j * k + (kk0 + t)];
                        }
                        out[ii * n + j] = sum;
                    }
                }
            }
            continue;
        }

        // Full panel: pack B^T(jj..jj+nr) for all k.
        if packed_b_full.len() != k * nr {
            packed_b_full.resize(k * nr, 0.0);
        }
        for kk0 in 0..k {
            let dst = &mut packed_b_full[kk0 * nr..(kk0 + 1) * nr];
            for j in 0..nr {
                dst[j] = b[(jj + j) * k + kk0];
            }
        }
        let packed_panel = &packed_b_full[..k * nr];

        if use_par {
            out.par_chunks_mut(chunk_elems)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let ii0 = chunk_idx * rows_per_chunk;
                    if ii0 >= m {
                        return;
                    }
                    let bm = ((ii0 + rows_per_chunk).min(m)) - ii0;

                    for kk0 in (0..k).step_by(kc_max) {
                        let kc = (k - kk0).min(kc_max);
                        let packed_b = &packed_panel[kk0 * nr..(kk0 + kc) * nr];

                        for i_local in (0..bm).step_by(MR) {
                            let mr = (bm - i_local).min(MR);
                            if mr == MR {
                                let a_ptr = &a[(ii0 + i_local) * k + kk0..];
                                let c_ptr = &mut out_chunk[i_local * n + jj..];
                                match mk {
                                    MicroKernel::Scalar4x4 if nr == 4 => {
                                        microkernel_scalar_4x4(a_ptr, k, packed_b, kc, c_ptr, n)
                                    }
                                    MicroKernel::Neon4x4 if nr == 4 => {
                                        #[cfg(target_arch = "aarch64")]
                                        unsafe {
                                            microkernel_neon_4x4(a_ptr, k, packed_b, kc, c_ptr, n)
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        microkernel_scalar_4x4(a_ptr, k, packed_b, kc, c_ptr, n)
                                    }
                                    MicroKernel::Avx2Fma4x8 if nr == 8 => {
                                        #[cfg(target_arch = "x86_64")]
                                        unsafe {
                                            microkernel_avx2_fma_4x8(
                                                a_ptr, k, packed_b, kc, c_ptr, n,
                                            )
                                        }
                                        #[cfg(not(target_arch = "x86_64"))]
                                        {
                                            for i in 0..MR {
                                                for j in 0..nr {
                                                    let mut sum = c_ptr[i * n + j];
                                                    for t in 0..kc {
                                                        sum +=
                                                            a_ptr[i * k + t] * packed_b[t * nr + j];
                                                    }
                                                    c_ptr[i * n + j] = sum;
                                                }
                                            }
                                        }
                                    }
                                    _ => {
                                        for i in 0..MR {
                                            for j in 0..nr {
                                                let mut sum = c_ptr[i * n + j];
                                                for t in 0..kc {
                                                    sum += a_ptr[i * k + t] * packed_b[t * nr + j];
                                                }
                                                c_ptr[i * n + j] = sum;
                                            }
                                        }
                                    }
                                }
                            } else {
                                for i in 0..mr {
                                    for j in 0..nr {
                                        let mut sum = out_chunk[(i_local + i) * n + (jj + j)];
                                        for t in 0..kc {
                                            sum += a[(ii0 + i_local + i) * k + (kk0 + t)]
                                                * b[(jj + j) * k + (kk0 + t)];
                                        }
                                        out_chunk[(i_local + i) * n + (jj + j)] = sum;
                                    }
                                }
                            }
                        }
                    }
                });
        } else {
            for kk0 in (0..k).step_by(kc_max) {
                let kc = (k - kk0).min(kc_max);
                let packed_b = &packed_panel[kk0 * nr..(kk0 + kc) * nr];

                for ii in (0..m).step_by(MR) {
                    let mr = (m - ii).min(MR);
                    if mr == MR {
                        let a_ptr = &a[ii * k + kk0..];
                        let c_ptr = &mut out[ii * n + jj..];
                        match mk {
                            MicroKernel::Scalar4x4 if nr == 4 => {
                                microkernel_scalar_4x4(a_ptr, k, packed_b, kc, c_ptr, n)
                            }
                            MicroKernel::Neon4x4 if nr == 4 => {
                                #[cfg(target_arch = "aarch64")]
                                unsafe {
                                    microkernel_neon_4x4(a_ptr, k, packed_b, kc, c_ptr, n)
                                }
                                #[cfg(not(target_arch = "aarch64"))]
                                microkernel_scalar_4x4(a_ptr, k, packed_b, kc, c_ptr, n)
                            }
                            MicroKernel::Avx2Fma4x8 if nr == 8 => {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    microkernel_avx2_fma_4x8(a_ptr, k, packed_b, kc, c_ptr, n)
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    for i in 0..MR {
                                        for j in 0..nr {
                                            let mut sum = c_ptr[i * n + j];
                                            for t in 0..kc {
                                                sum += a_ptr[i * k + t] * packed_b[t * nr + j];
                                            }
                                            c_ptr[i * n + j] = sum;
                                        }
                                    }
                                }
                            }
                            _ => {
                                for i in 0..MR {
                                    for j in 0..nr {
                                        let mut sum = c_ptr[i * n + j];
                                        for t in 0..kc {
                                            sum += a_ptr[i * k + t] * packed_b[t * nr + j];
                                        }
                                        c_ptr[i * n + j] = sum;
                                    }
                                }
                            }
                        }
                    } else {
                        for i in 0..mr {
                            for j in 0..nr {
                                let mut sum = out[(ii + i) * n + (jj + j)];
                                for t in 0..kc {
                                    sum +=
                                        a[(ii + i) * k + (kk0 + t)] * b[(jj + j) * k + (kk0 + t)];
                                }
                                out[(ii + i) * n + (jj + j)] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn matmul_tiled_tn_packed_add_into(
    a: &[f32],
    m: usize,
    n: usize,
    b: &[f32],
    p: usize,
    out: &mut [f32],
    block: usize,
) {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), m * p);
    debug_assert_eq!(out.len(), n * p);

    let mk = pick_microkernel();
    let nr = mk.nr();
    let kc_max = block.max(1);

    // Packed kernels tend to have higher per-task overhead (packing, microkernel setup),
    // so use a higher threshold before turning on Rayon.
    let use_par = parallel_enabled() && n * p >= par_min_elems().saturating_mul(8);
    // Use coarser chunks than `block` to keep Rayon overhead amortized.
    let rows_per_chunk = block.saturating_mul(4).max(MR).min(n).max(1);
    let chunk_elems = rows_per_chunk * p;

    let mut packed_b = vec![0.0f32; kc_max * nr];

    for jj in (0..p).step_by(nr) {
        let nr_eff = (p - jj).min(nr);
        for kk0 in (0..m).step_by(kc_max) {
            let kc = (m - kk0).min(kc_max);
            if kc * nr > packed_b.len() {
                packed_b.resize(kc * nr, 0.0);
            }
            pack_b_panel_nn(b, m, p, kk0, jj, kc, nr, &mut packed_b[..kc * nr]);
            let packed_b_block = &packed_b[..kc * nr];

            if use_par {
                out.par_chunks_mut(chunk_elems)
                    .enumerate()
                    .for_each(|(chunk_idx, out_chunk)| {
                        let ii0 = chunk_idx * rows_per_chunk;
                        if ii0 >= n {
                            return;
                        }
                        let bn = ((ii0 + rows_per_chunk).min(n)) - ii0;

                        let mut packed_a = vec![0.0f32; MR * kc_max];

                        for i_local in (0..bn).step_by(MR) {
                            let mr = (bn - i_local).min(MR);
                            let i0 = ii0 + i_local;
                            if mr == MR && nr_eff == nr {
                                if mr * kc > packed_a.len() {
                                    packed_a.resize(mr * kc, 0.0);
                                }
                                pack_a_panel_tn(a, m, n, i0, kk0, mr, kc, &mut packed_a[..mr * kc]);

                                let c_ptr = &mut out_chunk[i_local * p + jj..];
                                match mk {
                                    MicroKernel::Scalar4x4 if nr == 4 => microkernel_scalar_4x4(
                                        &packed_a[..mr * kc],
                                        kc,
                                        packed_b_block,
                                        kc,
                                        c_ptr,
                                        p,
                                    ),
                                    MicroKernel::Neon4x4 if nr == 4 => {
                                        #[cfg(target_arch = "aarch64")]
                                        unsafe {
                                            microkernel_neon_4x4(
                                                &packed_a[..mr * kc],
                                                kc,
                                                packed_b_block,
                                                kc,
                                                c_ptr,
                                                p,
                                            )
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        microkernel_scalar_4x4(
                                            &packed_a[..mr * kc],
                                            kc,
                                            packed_b_block,
                                            kc,
                                            c_ptr,
                                            p,
                                        )
                                    }
                                    MicroKernel::Avx2Fma4x8 if nr == 8 => {
                                        #[cfg(target_arch = "x86_64")]
                                        unsafe {
                                            microkernel_avx2_fma_4x8(
                                                &packed_a[..mr * kc],
                                                kc,
                                                packed_b_block,
                                                kc,
                                                c_ptr,
                                                p,
                                            )
                                        }
                                        #[cfg(not(target_arch = "x86_64"))]
                                        {
                                            for i in 0..MR {
                                                for j in 0..nr {
                                                    let mut sum = c_ptr[i * p + j];
                                                    for t in 0..kc {
                                                        sum += packed_a[i * kc + t]
                                                            * packed_b_block[t * nr + j];
                                                    }
                                                    c_ptr[i * p + j] = sum;
                                                }
                                            }
                                        }
                                    }
                                    _ => {
                                        for i in 0..MR {
                                            for j in 0..nr {
                                                let mut sum = c_ptr[i * p + j];
                                                for t in 0..kc {
                                                    sum += packed_a[i * kc + t]
                                                        * packed_b_block[t * nr + j];
                                                }
                                                c_ptr[i * p + j] = sum;
                                            }
                                        }
                                    }
                                }
                            } else {
                                for i in 0..mr {
                                    for j in 0..nr_eff {
                                        let mut sum = out_chunk[(i_local + i) * p + (jj + j)];
                                        for t in 0..kc {
                                            sum += a[(kk0 + t) * n + (i0 + i)]
                                                * b[(kk0 + t) * p + (jj + j)];
                                        }
                                        out_chunk[(i_local + i) * p + (jj + j)] = sum;
                                    }
                                }
                            }
                        }
                    });
            } else {
                let mut packed_a = vec![0.0f32; MR * kc_max];
                for ii in (0..n).step_by(MR) {
                    let mr = (n - ii).min(MR);
                    if mr == MR && nr_eff == nr {
                        if mr * kc > packed_a.len() {
                            packed_a.resize(mr * kc, 0.0);
                        }
                        pack_a_panel_tn(a, m, n, ii, kk0, mr, kc, &mut packed_a[..mr * kc]);

                        let c_ptr = &mut out[ii * p + jj..];
                        match mk {
                            MicroKernel::Scalar4x4 if nr == 4 => microkernel_scalar_4x4(
                                &packed_a[..mr * kc],
                                kc,
                                packed_b_block,
                                kc,
                                c_ptr,
                                p,
                            ),
                            MicroKernel::Neon4x4 if nr == 4 => {
                                #[cfg(target_arch = "aarch64")]
                                unsafe {
                                    microkernel_neon_4x4(
                                        &packed_a[..mr * kc],
                                        kc,
                                        packed_b_block,
                                        kc,
                                        c_ptr,
                                        p,
                                    )
                                }
                                #[cfg(not(target_arch = "aarch64"))]
                                microkernel_scalar_4x4(
                                    &packed_a[..mr * kc],
                                    kc,
                                    packed_b_block,
                                    kc,
                                    c_ptr,
                                    p,
                                )
                            }
                            MicroKernel::Avx2Fma4x8 if nr == 8 => {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    microkernel_avx2_fma_4x8(
                                        &packed_a[..mr * kc],
                                        kc,
                                        packed_b_block,
                                        kc,
                                        c_ptr,
                                        p,
                                    )
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                {
                                    for i in 0..MR {
                                        for j in 0..nr {
                                            let mut sum = c_ptr[i * p + j];
                                            for t in 0..kc {
                                                sum += packed_a[i * kc + t]
                                                    * packed_b_block[t * nr + j];
                                            }
                                            c_ptr[i * p + j] = sum;
                                        }
                                    }
                                }
                            }
                            _ => {
                                for i in 0..MR {
                                    for j in 0..nr {
                                        let mut sum = c_ptr[i * p + j];
                                        for t in 0..kc {
                                            sum +=
                                                packed_a[i * kc + t] * packed_b_block[t * nr + j];
                                        }
                                        c_ptr[i * p + j] = sum;
                                    }
                                }
                            }
                        }
                    } else {
                        for i in 0..mr {
                            for j in 0..nr_eff {
                                let mut sum = out[(ii + i) * p + (jj + j)];
                                for t in 0..kc {
                                    sum +=
                                        a[(kk0 + t) * n + (ii + i)] * b[(kk0 + t) * p + (jj + j)];
                                }
                                out[(ii + i) * p + (jj + j)] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
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

    matmul_tiled_into_slices_scalar(a, m, n, b, p, out, block);
}

fn matmul_tiled_into_slices_scalar(
    a: &[f32],
    m: usize,
    n: usize,
    b: &[f32],
    p: usize,
    out: &mut [f32],
    block: usize,
) {
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

pub fn matmul_tiled_packed_into_slices(
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
    assert!(block > 0);

    // Packed path (B-panel packing + microkernel) for improved cache locality.
    // Falls back to the scalar tiled loops for small problems to avoid packing overhead.
    if m.saturating_mul(n).saturating_mul(p) >= 64 * 64 * 64 {
        matmul_tiled_nn_packed_add_into(a, m, n, b, p, out, block);
    } else {
        matmul_tiled_into_slices_scalar(a, m, n, b, p, out, block);
    }
}

pub fn matmul_tiled_into(a: &Tensor, b: &Tensor, out: &mut [f32], block: usize) {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);
    matmul_tiled_into_slices(&a.data, m, n, &b.data, p, out, block);
}

pub fn matmul_tiled_packed_into(a: &Tensor, b: &Tensor, out: &mut [f32], block: usize) {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);
    matmul_tiled_packed_into_slices(&a.data, m, n, &b.data, p, out, block);
}

pub fn matmul_tiled(a: &Tensor, b: &Tensor, block: usize) -> Tensor {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);

    let mut result = vec![0.0; m * p];
    matmul_tiled_into(a, b, &mut result, block);

    Tensor::new(result, vec![m, p], a.requires_grad || b.requires_grad)
}

pub fn matmul_tiled_packed(a: &Tensor, b: &Tensor, block: usize) -> Tensor {
    let (m, n) = (a.shape[0], a.shape[1]);
    let (n2, p) = (b.shape[0], b.shape[1]);

    assert_eq!(n, n2);

    let mut result = vec![0.0; m * p];
    matmul_tiled_packed_into(a, b, &mut result, block);

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

    // Packed path for NT (A @ B^T) used heavily in MatMul backward.
    if block > 0 && m.saturating_mul(k).saturating_mul(n) >= 64 * 64 * 64 {
        matmul_tiled_nt_packed_add_into(a, m, k, b, n, out, block);
        return;
    }

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

    // Packed path for TN (A^T @ B) used heavily in MatMul backward.
    if block > 0 && m.saturating_mul(n).saturating_mul(p) >= 64 * 64 * 64 {
        matmul_tiled_tn_packed_add_into(a, m, n, b, p, out, block);
        return;
    }

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
