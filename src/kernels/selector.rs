#![allow(dead_code)]

use crate::tensor::tensor::Tensor;
use crate::kernels::naive::matmul_naive;
use crate::kernels::tiled::matmul_tiled;
use crate::kernels::tiled_mp::matmul_tiled_mp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    Naive,
    Tiled,
    TiledMP,
}

// Encodes measured/assumed performance knowledge into system behavior.
// `size` is the characteristic problem size (e.g. square matrix dimension).
pub fn select_kernel(size: usize) -> KernelType {
    // Data-driven thresholds (empirically tuned). Defaults can be overridden via env:
    // - POOLGRAD_SCHED_NAIVE_MAX (default 8)
    // - POOLGRAD_SCHED_TILED_MAX (default 8)
    // Block-level MP is only attempted inside the tiled-MP kernel.
    let naive_max = std::env::var("POOLGRAD_SCHED_NAIVE_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let tiled_max = std::env::var("POOLGRAD_SCHED_TILED_MAX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);

    if size <= naive_max {
        KernelType::Naive
    } else if size <= tiled_max {
        KernelType::Tiled
    } else {
        KernelType::TiledMP
    }
}

pub fn matmul(
    a: &Tensor,
    b: &Tensor,
    kernel: KernelType,
) -> Tensor {
    match kernel {
        KernelType::Naive => matmul_naive(a, b),
        KernelType::Tiled => matmul_tiled(a, b, 16),
        KernelType::TiledMP => matmul_tiled_mp(a, b, 16),
    }
}