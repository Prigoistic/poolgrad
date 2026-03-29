#![allow(dead_code)]

use crate::kernels::naive::matmul_naive;
use crate::kernels::naive::matmul_naive_add_into_slices_a_transposed;
use crate::kernels::naive::matmul_naive_add_into_slices_b_transposed;
use crate::kernels::naive::matmul_naive_into;
use crate::kernels::tiled::matmul_tiled;
use crate::kernels::tiled::matmul_tiled_add_into_slices_a_transposed;
use crate::kernels::tiled::matmul_tiled_add_into_slices_b_transposed;
use crate::kernels::tiled::matmul_tiled_into;
use crate::kernels::tiled::matmul_tiled_packed;
use crate::kernels::tiled::matmul_tiled_packed_into;
use crate::kernels::tiled_mp::matmul_tiled_mp;
use crate::kernels::tiled_mp::matmul_tiled_mp_add_into_slices_a_transposed;
use crate::kernels::tiled_mp::matmul_tiled_mp_add_into_slices_b_transposed;
use crate::kernels::tiled_mp::matmul_tiled_mp_into;
use crate::tensor::tensor::Tensor;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    Naive,
    Tiled,
    TiledPacked,
    TiledMP,
}

fn parallel_enabled() -> bool {
    static PAR_ENABLED: OnceLock<bool> = OnceLock::new();
    *PAR_ENABLED.get_or_init(|| {
        std::env::var("POOLGRAD_PAR")
            .ok()
            .as_deref()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true)
    })
}

#[derive(Debug, Clone, Copy)]
struct SchedulerConfig {
    naive_max: usize,
    tiled_max: usize,
    packed_max: usize,
}

fn scheduler_config() -> &'static SchedulerConfig {
    static CONFIG: OnceLock<SchedulerConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        // Data-driven thresholds (empirically tuned). Defaults can be overridden via env:
        // - POOLGRAD_SCHED_NAIVE_MAX (default 8)
        // - POOLGRAD_SCHED_TILED_MAX (default 8)
        // - POOLGRAD_SCHED_PACKED_MAX (default 512)
        // Block-level MP is only attempted inside the tiled-MP kernel.
        let (default_naive_max, default_tiled_max) = if parallel_enabled() {
            // Empirically, the naive kernel is best only for small sizes.
            // Past that, the packed+microkernel path tends to dominate.
            (32usize, 32usize)
        } else {
            (8usize, 8usize)
        };

        let default_packed_max = if parallel_enabled() {
            2048usize
        } else {
            512usize
        };

        let naive_max = std::env::var("POOLGRAD_SCHED_NAIVE_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_naive_max);
        let tiled_max = std::env::var("POOLGRAD_SCHED_TILED_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_tiled_max);

        let packed_max = std::env::var("POOLGRAD_SCHED_PACKED_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_packed_max);

        SchedulerConfig {
            naive_max,
            tiled_max,
            packed_max,
        }
    })
}

static KERNEL_PROFILE: OnceLock<Option<HashMap<usize, KernelType>>> = OnceLock::new();

fn parse_kernel_name(s: &str) -> Option<KernelType> {
    match s.trim() {
        "Naive" | "naive" => Some(KernelType::Naive),
        "Tiled" | "tiled" => Some(KernelType::Tiled),
        "TiledPacked" | "tiledpacked" | "tiled_packed" | "tiled-packed" => {
            Some(KernelType::TiledPacked)
        }
        "TiledMP" | "tiledmp" | "tiled_mp" | "tiled-mp" => Some(KernelType::TiledMP),
        _ => None,
    }
}

fn forced_kernel() -> Option<KernelType> {
    static FORCED: OnceLock<Option<KernelType>> = OnceLock::new();
    *FORCED.get_or_init(|| {
        let v = std::env::var("POOLGRAD_FORCE_KERNEL").ok()?;
        parse_kernel_name(&v)
    })
}

fn load_kernel_profile_once() -> &'static Option<HashMap<usize, KernelType>> {
    KERNEL_PROFILE.get_or_init(|| {
        let path = std::env::var("POOLGRAD_KERNEL_PROFILE").ok()?;
        let text = std::fs::read_to_string(&path).ok()?;
        let mut map = HashMap::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let mut it = line.split_whitespace();
            let size_s = it.next();
            let kernel_s = it.next();
            if size_s.is_none() || kernel_s.is_none() {
                continue;
            }

            let size: usize = match size_s.unwrap().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let kernel = match parse_kernel_name(kernel_s.unwrap()) {
                Some(k) => k,
                None => continue,
            };
            map.insert(size, kernel);
        }

        Some(map)
    })
}

// Encodes measured/assumed performance knowledge into system behavior.
// `size` is the characteristic problem size (e.g. square matrix dimension).
pub fn select_kernel(size: usize) -> KernelType {
    if let Some(k) = forced_kernel() {
        return k;
    }

    if let Some(profile) = load_kernel_profile_once().as_ref()
        && let Some(&k) = profile.get(&size)
    {
        return k;
    }

    let cfg = scheduler_config();

    if size <= cfg.naive_max {
        KernelType::Naive
    } else if size <= cfg.tiled_max {
        KernelType::Tiled
    } else if size <= cfg.packed_max {
        KernelType::TiledPacked
    } else {
        KernelType::TiledMP
    }
}

/// Shape-aware kernel selection.
///
/// This is intended for real workloads where matrices are not necessarily square.
/// Benchmarks/profiles may continue using `select_kernel(size)`.
pub fn select_kernel_mm(m: usize, n: usize, p: usize) -> KernelType {
    let size_hint = m.max(n).max(p);

    if let Some(k) = forced_kernel() {
        return k;
    }

    // Preserve existing size-based profile override for compatibility.
    if let Some(profile) = load_kernel_profile_once().as_ref()
        && let Some(&k) = profile.get(&size_hint)
    {
        return k;
    }

    let cfg = scheduler_config();

    let min_dim = m.min(n).min(p);
    let work = m.saturating_mul(n).saturating_mul(p);

    // If any dimension is extremely small, packing overhead usually dominates.
    if size_hint <= cfg.naive_max || min_dim <= 4 {
        return KernelType::Naive;
    }

    // Avoid packing for small total work.
    let tiny_work_max = std::env::var("POOLGRAD_SCHED_TINY_WORK_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64 * 64 * 64);
    if work < tiny_work_max {
        return KernelType::Tiled;
    }

    // If the reduction dimension is small, the scalar tiled kernel tends to do fine.
    // This also avoids packing when `n` is skinny.
    let inner_small_max = std::env::var("POOLGRAD_SCHED_INNER_SMALL_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(32);
    if n <= inner_small_max {
        return KernelType::Tiled;
    }

    if size_hint <= cfg.tiled_max {
        KernelType::Tiled
    } else if size_hint <= cfg.packed_max {
        KernelType::TiledPacked
    } else {
        KernelType::TiledMP
    }
}

/// WARNING: does NOT build an autograd graph.
pub fn matmul_no_grad(a: &Tensor, b: &Tensor, kernel: KernelType) -> Tensor {
    match kernel {
        KernelType::Naive => matmul_naive(a, b),
        KernelType::Tiled => matmul_tiled(a, b, 16),
        KernelType::TiledPacked => matmul_tiled_packed(a, b, 16),
        KernelType::TiledMP => matmul_tiled_mp(a, b, 16),
    }
}

#[deprecated(
    note = "selector::matmul does not build an autograd graph; use matmul_no_grad explicitly"
)]
pub fn matmul(a: &Tensor, b: &Tensor, kernel: KernelType) -> Tensor {
    matmul_no_grad(a, b, kernel)
}

pub fn matmul_into(a: &Tensor, b: &Tensor, kernel: KernelType, out: &mut [f32]) {
    match kernel {
        KernelType::Naive => matmul_naive_into(a, b, out),
        KernelType::Tiled => matmul_tiled_into(a, b, out, 16),
        KernelType::TiledPacked => matmul_tiled_packed_into(a, b, out, 16),
        KernelType::TiledMP => matmul_tiled_mp_into(a, b, out, 16),
    }
}

pub fn matmul_add_into_slices_b_transposed(
    kernel: KernelType,
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
) {
    match kernel {
        KernelType::Naive => matmul_naive_add_into_slices_b_transposed(a, m, k, b, n, out),
        KernelType::Tiled | KernelType::TiledPacked => {
            matmul_tiled_add_into_slices_b_transposed(a, m, k, b, n, out, 16)
        }
        KernelType::TiledMP => matmul_tiled_mp_add_into_slices_b_transposed(a, m, k, b, n, out, 16),
    }
}

pub fn matmul_add_into_slices_a_transposed(
    kernel: KernelType,
    a: &[f32],
    m: usize,
    n: usize,
    b: &[f32],
    p: usize,
    out: &mut [f32],
) {
    match kernel {
        KernelType::Naive => matmul_naive_add_into_slices_a_transposed(a, m, n, b, p, out),
        KernelType::Tiled | KernelType::TiledPacked => {
            matmul_tiled_add_into_slices_a_transposed(a, m, n, b, p, out, 16)
        }
        KernelType::TiledMP => matmul_tiled_mp_add_into_slices_a_transposed(a, m, n, b, p, out, 16),
    }
}

#[cfg(test)]
mod tests {
    use super::{KernelType, matmul_into};
    use super::{matmul_add_into_slices_a_transposed, matmul_add_into_slices_b_transposed};
    use crate::tensor::tensor::Tensor;

    fn ref_mm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
        // (m x k) @ (k x n) -> (m x n)
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for t in 0..k {
                    acc += a[i * k + t] * b[t * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn ref_mm_b_t(a: &[f32], m: usize, k: usize, b_t: &[f32], n: usize) -> Vec<f32> {
        // (m x k) @ (n x k)^T -> (m x n)
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for t in 0..k {
                    acc += a[i * k + t] * b_t[j * k + t];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    fn ref_mm_a_t(a: &[f32], m: usize, n: usize, b: &[f32], p: usize) -> Vec<f32> {
        // (m x n)^T @ (m x p) -> (n x p)
        let mut out = vec![0.0; n * p];
        for i in 0..n {
            for j in 0..p {
                let mut acc = 0.0;
                for t in 0..m {
                    acc += a[t * n + i] * b[t * p + j];
                }
                out[i * p + j] = acc;
            }
        }
        out
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut maxd = 0.0f32;
        for i in 0..a.len() {
            let d = (a[i] - b[i]).abs();
            if d > maxd {
                maxd = d;
            }
        }
        maxd
    }

    #[test]
    fn kernels_support_nn_nt_tn_and_match_reference() {
        let m = 3usize;
        let k = 5usize;
        let n = 4usize;
        let p = 6usize;

        // Deterministic small matrices (avoid RNG dependency).
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1 - 0.7).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * -0.07 + 0.3).collect();
        let b_t: Vec<f32> = (0..(n * k)).map(|i| (i as f32) * 0.05 - 0.2).collect();
        let b2: Vec<f32> = (0..(m * p)).map(|i| (i as f32) * 0.03 + 0.1).collect();

        let a_ten = Tensor::new(a.clone(), vec![m, k], false);
        let b_ten = Tensor::new(b.clone(), vec![k, n], false);

        let ref_nn = ref_mm(&a, m, k, &b, n);
        let ref_nt = ref_mm_b_t(&a, m, k, &b_t, n);
        let ref_tn = ref_mm_a_t(&a, m, k, &b2, p);

        for kernel in [
            KernelType::Naive,
            KernelType::Tiled,
            KernelType::TiledPacked,
            KernelType::TiledMP,
        ] {
            // NN
            let mut out = vec![0.0; m * n];
            matmul_into(&a_ten, &b_ten, kernel, &mut out);
            assert!(
                max_abs_diff(&out, &ref_nn) < 1e-4,
                "NN mismatch for {:?}",
                kernel
            );

            // NT: out += A @ B^T
            let mut out_nt = vec![0.0; m * n];
            matmul_add_into_slices_b_transposed(kernel, &a, m, k, &b_t, n, &mut out_nt);
            assert!(
                max_abs_diff(&out_nt, &ref_nt) < 1e-4,
                "NT mismatch for {:?}",
                kernel
            );

            // TN: out += A^T @ B
            let mut out_tn = vec![0.0; k * p];
            matmul_add_into_slices_a_transposed(kernel, &a, m, k, &b2, p, &mut out_tn);
            assert!(
                max_abs_diff(&out_tn, &ref_tn) < 1e-4,
                "TN mismatch for {:?}",
                kernel
            );
        }
    }
}
