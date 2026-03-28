#![allow(dead_code)]

use crate::kernels::naive::matmul_naive;
use crate::kernels::tiled::matmul_tiled;
use crate::kernels::tiled_mp::matmul_tiled_mp;
use crate::tensor::tensor::Tensor;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    Naive,
    Tiled,
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
}

fn scheduler_config() -> &'static SchedulerConfig {
    static CONFIG: OnceLock<SchedulerConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        // Data-driven thresholds (empirically tuned). Defaults can be overridden via env:
        // - POOLGRAD_SCHED_NAIVE_MAX (default 8)
        // - POOLGRAD_SCHED_TILED_MAX (default 8)
        // Block-level MP is only attempted inside the tiled-MP kernel.
        let (default_naive_max, default_tiled_max) = if parallel_enabled() {
            // With Rayon enabled, naive (parallel over rows) is often competitive well past tiny sizes.
            (256usize, 256usize)
        } else {
            (8usize, 8usize)
        };

        let naive_max = std::env::var("POOLGRAD_SCHED_NAIVE_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_naive_max);
        let tiled_max = std::env::var("POOLGRAD_SCHED_TILED_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(default_tiled_max);

        SchedulerConfig {
            naive_max,
            tiled_max,
        }
    })
}

static KERNEL_PROFILE: OnceLock<Option<HashMap<usize, KernelType>>> = OnceLock::new();

fn parse_kernel_name(s: &str) -> Option<KernelType> {
    match s.trim() {
        "Naive" | "naive" => Some(KernelType::Naive),
        "Tiled" | "tiled" => Some(KernelType::Tiled),
        "TiledMP" | "tiledmp" | "tiled_mp" | "tiled-mp" => Some(KernelType::TiledMP),
        _ => None,
    }
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
    } else {
        KernelType::TiledMP
    }
}

pub fn matmul(a: &Tensor, b: &Tensor, kernel: KernelType) -> Tensor {
    match kernel {
        KernelType::Naive => matmul_naive(a, b),
        KernelType::Tiled => matmul_tiled(a, b, 16),
        KernelType::TiledMP => matmul_tiled_mp(a, b, 16),
    }
}
