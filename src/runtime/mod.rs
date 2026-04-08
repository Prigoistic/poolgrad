use std::time::Instant;

use crate::autograd::graph::Graph;
use crate::kernels::naive::matmul_naive;
use crate::kernels::selector::select_kernel_mm;
use crate::kernels::selector::{KernelType, matmul_no_grad as matmul_by_kernel};
use crate::kernels::tiled::{matmul_tiled, matmul_tiled_packed};
use crate::memory::pool::MemoryPool;
use crate::nn::linear::Linear;
use crate::nn::loss::mse;
use crate::tensor::store::TensorStore;
use crate::tensor::tensor::Tensor;

fn install_broken_pipe_hook() {
    // When piping to `head`/`awk`, the reader may close stdout early.
    // Rust's println! will then panic with "failed printing to stdout: Broken pipe".
    // Treat that as a normal early-exit instead of a crash.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let mut is_broken_pipe = false;

        if let Some(s) = info.payload().downcast_ref::<&str>() {
            is_broken_pipe = s.contains("Broken pipe") || s.contains("failed printing to stdout");
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            is_broken_pipe = s.contains("Broken pipe") || s.contains("failed printing to stdout");
        }

        if is_broken_pipe {
            std::process::exit(0);
        }

        default_hook(info);
    }));
}

struct Config {
    use_pool: bool,
    verbose: bool,
}

fn run_training(config: &Config) -> MemoryPool {
    let mut store = TensorStore::new();
    let mut graph = Graph::new();
    let mut pool = MemoryPool::new();
    pool.enabled = config.use_pool;
    let mut x_id = store.add(Tensor::new(vec![1.0, 2.0], vec![1, 2], true));
    let mut target_id = store.add(Tensor::new(vec![3.0, 3.0, 3.0], vec![1, 3], false));

    let mut linear = Linear::new(2, 3, &mut store);

    for epoch in 0..10 {
        // Bounded memory: we keep only persistent tensors in the store across epochs.
        // (inputs + targets + parameters)
        debug_assert_eq!(
            store.tensors.len(),
            4,
            "TensorStore grew across epochs; expected only persistent tensors"
        );

        // forward
        let out_id = linear.forward(x_id, &mut store, &mut graph, &mut pool);
        let loss_id = mse(out_id, target_id, &mut store, &mut graph, &mut pool);

        // backward
        if config.use_pool {
            graph.backward(&mut store, loss_id, &mut pool);
        } else {
            // When pooling is disabled, run the deterministic planned backward.
            // This avoids pool get/release calls entirely.
            let _planner = graph.backward_planned(&mut store, loss_id);
        }

        // update weights
        let lr = 0.01;

        let weight = store.get_mut(linear.weight_id);
        for i in 0..weight.data.len() {
            weight.data[i] -= lr * weight.grad[i];
        }

        let bias = store.get_mut(linear.bias_id);
        for i in 0..bias.data.len() {
            bias.data[i] -= lr * bias.grad[i];
        }

        if config.verbose {
            println!(
                "Epoch {} Loss: {:?}, Output: {:?}",
                epoch,
                store.get(loss_id).data,
                store.get(out_id).data
            );
        }

        // CRITICAL: Rebuild store+graph each epoch to avoid unbounded TensorStore growth.
        // Keep the persistent tensors (x, target, weights, bias) and drop ephemerals.
        let mut old = std::mem::take(&mut store.tensors);

        // Extract persistent tensors by id. Remove in descending order to keep indices valid.
        let mut bias = old.swap_remove(linear.bias_id);
        let mut weight = old.swap_remove(linear.weight_id);
        let mut target = old.swap_remove(target_id);
        let mut x = old.swap_remove(x_id);

        // Ensure persistent tensors look like true leafs.
        x.creator = None;
        target.creator = None;
        weight.creator = None;
        bias.creator = None;

        x.zero_grad();
        weight.zero_grad();
        bias.zero_grad();

        drop(old);

        store = TensorStore::new();
        graph = Graph::new();

        x_id = store.add(x);
        target_id = store.add(target);
        linear.weight_id = store.add(weight);
        linear.bias_id = store.add(bias);
    }

    pool
}

fn print_metrics(label: &str, pool: &MemoryPool) {
    let total = (pool.allocations + pool.reuses) as f32;
    let reuse_rate = if total > 0.0 {
        (pool.reuses as f32 / total) * 100.0
    } else {
        0.0
    };

    let live_peak_bytes = pool.peak_memory;
    let cached_peak_bytes = pool.cached_peak;
    let resident_peak_bytes = pool.resident_peak;

    println!("=== {} ===", label);
    println!(
        "Mode | Allocations | Reuses | ReuseRate | LivePeak | CachedPeak | ResidentPeak\n{} | {} | {} | {:.2}% | {} | {} | {}",
        label,
        pool.allocations,
        pool.reuses,
        reuse_rate,
        live_peak_bytes,
        cached_peak_bytes,
        resident_peak_bytes
    );
}

#[derive(Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        // Deterministic float in [-1, 1).
        let u = self.next_u64();
        let v = (u >> 40) as u32; // 24-ish bits
        let f = (v as f32) / (u32::MAX as f32);
        2.0 * f - 1.0
    }
}

fn random_matrix(rng: &mut XorShift64, m: usize, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(m * n);
    for _ in 0..m * n {
        v.push(rng.next_f32());
    }
    v
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

fn validate_kernels() {
    let seed = 0xC0FFEE_u64;
    let mut rng = XorShift64::new(seed);
    let sizes = [1usize, 2, 3, 4, 5, 8, 16, 31, 32, 63, 64, 96, 128, 256];

    println!(
        "\nCorrectness validation (seed={:#x}; tol: 1e-4 (Tiled/TiledPacked), 1e-3 (TiledMP))",
        seed
    );
    println!("Size | Kernel  | MaxAbsErr");
    println!("-----|---------|----------");

    for &size in &sizes {
        let a = Tensor::new(random_matrix(&mut rng, size, size), vec![size, size], false);
        let b = Tensor::new(random_matrix(&mut rng, size, size), vec![size, size], false);

        let reference = matmul_naive(&a, &b);

        for (kernel, name) in [
            (KernelType::Tiled, "Tiled"),
            (KernelType::TiledPacked, "TiledPk"),
            (KernelType::TiledMP, "TiledMP"),
        ] {
            let out = matmul_by_kernel(&a, &b, kernel);
            let err = max_abs_diff(&reference.data, &out.data);
            let tol = match kernel {
                KernelType::TiledMP => 1e-3,
                _ => 1e-4,
            };
            println!("{:>4} | {:<7} | {:>8.2e}", size, name, err);
            assert!(
                err < tol,
                "kernel {:?} failed validation at size {}: max_abs_err={} (tol={})",
                kernel,
                size,
                err,
                tol
            );
        }
    }
}

fn bench_ms<F: FnMut()>(mut f: F, warmup: usize, trials: usize) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }

    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = samples[samples.len() / 2];
    let p95_idx = (samples.len() * 95)
        .div_ceil(100)
        .saturating_sub(1)
        .min(samples.len() - 1);
    let p95 = samples[p95_idx];

    (median, p95)
}

fn run_kernel_benchmark() {
    let seed = 0xBADC0DE_u64;
    let mut rng = XorShift64::new(seed);
    // Keep this list in sync with the README experiment section.
    let sizes = [32usize, 64, 128, 256, 512];

    let warmup = std::env::var("POOLGRAD_BENCH_WARMUP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2);
    let trials = std::env::var("POOLGRAD_BENCH_TRIALS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(9);

    println!(
        "\nKernel benchmark (seed={:#x}, warmup={}, trials={})",
        seed, warmup, trials
    );
    println!(
        "Size | Naive_med | Naive_p95 | Tiled_med | Tiled_p95 | TPk_med  | TPk_p95  | TiledMP_med | TiledMP_p95 | Winner   | Scheduler"
    );
    println!(
        "-----|----------|----------|----------|----------|---------|---------|------------|------------|----------|----------"
    );

    let mut winners: Vec<(usize, KernelType)> = Vec::new();

    for &size in &sizes {
        let a = Tensor::new(random_matrix(&mut rng, size, size), vec![size, size], false);
        let b = Tensor::new(random_matrix(&mut rng, size, size), vec![size, size], false);

        let reference = matmul_naive(&a, &b);

        let (naive_med, naive_p95) = bench_ms(
            || {
                let _ = matmul_naive(&a, &b);
            },
            warmup,
            trials,
        );

        let tiled_out = matmul_tiled(&a, &b, 16);
        let tiled_err = max_abs_diff(&reference.data, &tiled_out.data);
        assert!(
            tiled_err < 1e-4,
            "Tiled failed validation at size {}: err={}",
            size,
            tiled_err
        );
        let (tiled_med, tiled_p95) = bench_ms(
            || {
                let _ = matmul_tiled(&a, &b, 16);
            },
            warmup,
            trials,
        );

        let tiled_pk_out = matmul_tiled_packed(&a, &b, 16);
        let tiled_pk_err = max_abs_diff(&reference.data, &tiled_pk_out.data);
        assert!(
            tiled_pk_err < 1e-4,
            "TiledPacked failed validation at size {}: err={}",
            size,
            tiled_pk_err
        );
        let (tiled_pk_med, tiled_pk_p95) = bench_ms(
            || {
                let _ = matmul_tiled_packed(&a, &b, 16);
            },
            warmup,
            trials,
        );

        let tiled_mp_out = crate::kernels::tiled_mp::matmul_tiled_mp(&a, &b, 16);
        let tiled_mp_err = max_abs_diff(&reference.data, &tiled_mp_out.data);
        assert!(
            tiled_mp_err < 1e-3,
            "TiledMP failed validation at size {}: err={} (tol=1e-3)",
            size,
            tiled_mp_err
        );
        let (tiled_mp_med, tiled_mp_p95) = bench_ms(
            || {
                let _ = crate::kernels::tiled_mp::matmul_tiled_mp(&a, &b, 16);
            },
            warmup,
            trials,
        );

        let (winner_label, winner_kernel) =
            if naive_med <= tiled_med && naive_med <= tiled_pk_med && naive_med <= tiled_mp_med {
                ("Naive", KernelType::Naive)
            } else if tiled_med <= tiled_pk_med && tiled_med <= tiled_mp_med {
                ("Tiled", KernelType::Tiled)
            } else if tiled_pk_med <= tiled_mp_med {
                ("TiledPk", KernelType::TiledPacked)
            } else {
                ("TiledMP", KernelType::TiledMP)
            };

        let scheduled = select_kernel_mm(size, size, size);

        println!(
            "{:>4} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>7.3} | {:>7.3} | {:>10.3} | {:>10.3} | {:<8} | {:?}",
            size,
            naive_med,
            naive_p95,
            tiled_med,
            tiled_p95,
            tiled_pk_med,
            tiled_pk_p95,
            tiled_mp_med,
            tiled_mp_p95,
            winner_label,
            scheduled
        );

        winners.push((size, winner_kernel));
    }

    if let Ok(path) = std::env::var("POOLGRAD_KERNEL_PROFILE_WRITE") {
        use std::fmt::Write as _;
        let mut text = String::new();
        let _ = writeln!(&mut text, "# poolgrad kernel profile");
        let _ = writeln!(&mut text, "# format: <size> <kernel>");
        for (size, k) in &winners {
            let _ = writeln!(&mut text, "{} {:?}", size, k);
        }
        if let Err(e) = std::fs::write(&path, text) {
            eprintln!("Failed to write kernel profile to {}: {}", path, e);
        } else {
            println!("\nWrote kernel profile: {}", path);
        }
    }
}

fn run_forward_backward_step_benchmark() {
    use crate::memory::stats::{MemoryBytes, activation_bytes};
    use crate::memory::temp;
    use crate::tensor::tensor::matmul_with_pool;

    // Measures a training-style step: forward matmul + backward gradients.
    let sizes = [64usize, 128, 256];

    let warmup = std::env::var("POOLGRAD_BENCH_WARMUP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2);
    let trials = std::env::var("POOLGRAD_BENCH_TRIALS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(9);

    println!(
        "\nForward+Backward step benchmark (warmup={}, trials={})",
        warmup, trials
    );
    println!(
        "Size | Mode    | Kernel  | Step_med | Step_p95 | Allocs | GradReservedBytes | GradLiveBytes | ActivationBytes | TempBytes | TotalBytes"
    );
    println!(
        "-----|---------|---------|---------|---------|--------|------------------|-------------|----------------|----------|---------"
    );

    for &size in &sizes {
        let kernel = select_kernel_mm(size, size, size);

        for &(mode_name, use_planner) in &[("pool", false), ("plan", true)] {
            // Warmup
            for _ in 0..warmup {
                temp::reset();
                let mut store = TensorStore::new();
                let mut graph = Graph::new();
                let mut pool = MemoryPool::new();
                pool.enabled = true;

                let a_id = store.add(Tensor::new(vec![1.0; size * size], vec![size, size], true));
                let b_id = store.add(Tensor::new(vec![1.0; size * size], vec![size, size], true));
                let out_id = matmul_with_pool(a_id, b_id, &mut store, &mut graph, &mut pool);
                let seed = vec![1.0; store.get(out_id).data.len()];
                if use_planner {
                    let _ = graph.backward_seeded_planned(&mut store, out_id, &seed);
                } else {
                    graph.backward_seeded(&mut store, out_id, &seed, &mut pool);
                }
            }

            let mut samples = Vec::with_capacity(trials);
            let mut allocs_samples: Vec<usize> = Vec::with_capacity(trials);
            let mut grad_reserved_samples: Vec<usize> = Vec::with_capacity(trials);
            let mut grad_live_samples: Vec<usize> = Vec::with_capacity(trials);
            let mut activation_samples: Vec<usize> = Vec::with_capacity(trials);
            let mut temp_samples: Vec<usize> = Vec::with_capacity(trials);
            let mut total_samples: Vec<usize> = Vec::with_capacity(trials);

            for _ in 0..trials {
                temp::reset();
                let start = Instant::now();
                let mut store = TensorStore::new();
                let mut graph = Graph::new();
                let mut pool = MemoryPool::new();
                pool.enabled = true;

                let a_id = store.add(Tensor::new(vec![1.0; size * size], vec![size, size], true));
                let b_id = store.add(Tensor::new(vec![1.0; size * size], vec![size, size], true));
                let out_id = matmul_with_pool(a_id, b_id, &mut store, &mut graph, &mut pool);
                let seed = vec![1.0; store.get(out_id).data.len()];

                let (allocs, grad_reserved_bytes, grad_live_bytes) = if use_planner {
                    let planner = graph.backward_seeded_planned(&mut store, out_id, &seed);
                    (
                        planner.allocations,
                        planner.reserved_bytes(),
                        planner.peak_live_bytes,
                    )
                } else {
                    graph.backward_seeded(&mut store, out_id, &seed, &mut pool);
                    (pool.allocations, pool.resident_peak, pool.peak_memory)
                };

                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                samples.push(elapsed_ms);

                let activation = activation_bytes(&store);
                let temp_bytes = temp::peak_bytes();
                let mem = MemoryBytes {
                    grad_reserved_bytes,
                    grad_live_bytes,
                    activation_bytes: activation,
                    temp_bytes,
                };
                let total_bytes = mem.total_reserved_bytes();

                allocs_samples.push(allocs);
                grad_reserved_samples.push(mem.grad_reserved_bytes);
                grad_live_samples.push(mem.grad_live_bytes);
                activation_samples.push(mem.activation_bytes);
                temp_samples.push(mem.temp_bytes);
                total_samples.push(total_bytes);
            }

            let pick_med_p95 = |mut v: Vec<f64>| -> (f64, f64) {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = v[v.len() / 2];
                let p95_idx = (v.len() * 95)
                    .div_ceil(100)
                    .saturating_sub(1)
                    .min(v.len() - 1);
                let p95 = v[p95_idx];
                (med, p95)
            };

            let (median, p95) = pick_med_p95(samples);

            // For the memory/alloc columns, report maxima across trials (conservative).
            let allocs = *allocs_samples.iter().max().unwrap_or(&0);
            let grad_reserved_bytes = *grad_reserved_samples.iter().max().unwrap_or(&0);
            let grad_live_bytes = *grad_live_samples.iter().max().unwrap_or(&0);
            let activation_bytes = *activation_samples.iter().max().unwrap_or(&0);
            let temp_bytes = *temp_samples.iter().max().unwrap_or(&0);
            let total_bytes = *total_samples.iter().max().unwrap_or(&0);

            println!(
                "{:>4} | {:>7} | {:>7?} | {:>7.3} | {:>7.3} | {:>6} | {:>16} | {:>11} | {:>14} | {:>8} | {:>9}",
                size,
                mode_name,
                kernel,
                median,
                p95,
                allocs,
                grad_reserved_bytes,
                grad_live_bytes,
                activation_bytes,
                temp_bytes,
                total_bytes
            );
        }
    }
}

fn run_kernel_pool_interaction_experiment() -> usize {
    use crate::tensor::tensor::matmul_with_pool;

    // Keep this list in sync with the README experiment section.
    let sizes = [32usize, 64, 128, 256, 512];
    let mut pool = MemoryPool::new();
    pool.enabled = true;

    println!(
        "\nKernel+Pool interaction experiment (scheduler drives kernel, pool drives grad reuse)"
    );
    println!("Size | Kernel | Iters | Time (ms) | Alloc | Reuse | LivePeak | ResidentPeak");
    println!("-----|--------|-------|-----------|-------|-------|---------|------------");

    let mut exp_peak = 0usize;

    for &size in &sizes {
        let iters = if size <= 128 {
            10usize
        } else if size <= 256 {
            3usize
        } else {
            1usize
        };
        let kernel: KernelType = select_kernel_mm(size, size, size);

        let alloc0 = pool.allocations;
        let reuse0 = pool.reuses;

        let start = Instant::now();
        for _ in 0..iters {
            let mut store = TensorStore::new();
            let mut graph = Graph::new();

            let a_id = store.add(Tensor::new(vec![1.0; size * size], vec![size, size], true));
            let b_id = store.add(Tensor::new(vec![1.0; size * size], vec![size, size], true));

            let out_id = matmul_with_pool(a_id, b_id, &mut store, &mut graph, &mut pool);
            let seed = vec![1.0; store.get(out_id).data.len()];
            graph.backward_seeded(&mut store, out_id, &seed, &mut pool);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let alloc = pool.allocations - alloc0;
        let reuse = pool.reuses - reuse0;

        println!(
            "{:>4} | {:>6?} | {:>5} | {:>9.3} | {:>5} | {:>5} | {:>7} | {:>11}",
            size, kernel, iters, elapsed_ms, alloc, reuse, pool.peak_memory, pool.resident_peak
        );

        exp_peak = exp_peak.max(pool.peak_memory);
    }

    exp_peak
}

pub fn run() {
    install_broken_pipe_hook();

    // Ensure reproducible, end-to-end comparable metrics across runs.
    MemoryPool::reset_global_metrics();

    let verbose = std::env::var("POOLGRAD_VERBOSE").ok().as_deref() == Some("1");
    let baseline = Config {
        use_pool: false,
        verbose,
    };
    let pooled = Config {
        use_pool: true,
        verbose,
    };

    let baseline_pool = run_training(&baseline);
    let pooled_pool = run_training(&pooled);

    print_metrics("Baseline (pool OFF)", &baseline_pool);
    print_metrics("Pooled (pool ON)", &pooled_pool);

    validate_kernels();
    run_kernel_benchmark();
    run_forward_backward_step_benchmark();

    let exp_peak = run_kernel_pool_interaction_experiment();

    let global_peak = MemoryPool::global_peak_memory_bytes();
    println!("\nGlobal peak memory (bytes): {}", global_peak);
    println!("Interaction experiment peak (bytes): {}", exp_peak);
    println!(
        "Note: global peak tracks the max pool-bytes checked out across the entire process, \
including all benchmarks; the interaction experiment peak is scoped to that experiment only."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_lifecycle_is_bounded_across_epochs() {
        let mut store = TensorStore::new();
        let mut graph = Graph::new();
        let mut pool = MemoryPool::new();
        pool.enabled = true;

        let mut x_id = store.add(Tensor::new(vec![1.0, 2.0], vec![1, 2], true));
        let mut target_id = store.add(Tensor::new(vec![3.0, 3.0, 3.0], vec![1, 3], false));

        let mut linear = Linear::new(2, 3, &mut store);

        let mut cached_end: Vec<usize> = Vec::new();

        for _epoch in 0..6 {
            // The store should contain only the persistent tensors at the start of each epoch.
            assert_eq!(
                store.tensors.len(),
                4,
                "expected TensorStore to stay epoch-bounded"
            );

            // forward
            let out_id = linear.forward(x_id, &mut store, &mut graph, &mut pool);
            let loss_id = mse(out_id, target_id, &mut store, &mut graph, &mut pool);

            // backward
            graph.backward(&mut store, loss_id, &mut pool);

            // update weights
            let lr = 0.01;

            let weight = store.get_mut(linear.weight_id);
            for i in 0..weight.data.len() {
                weight.data[i] -= lr * weight.grad[i];
            }

            let bias = store.get_mut(linear.bias_id);
            for i in 0..bias.data.len() {
                bias.data[i] -= lr * bias.grad[i];
            }

            // Intermediate grad buffers are released during `backward`. The pool should only
            // have persistent leaf grads checked out (inputs/params).
            let persistent_bytes = store
                .tensors
                .iter()
                .filter(|t| t.creator.is_none() && t.requires_grad)
                .map(|t| t.data.len() * 4)
                .sum::<usize>();
            assert_eq!(
                pool.current_memory, persistent_bytes,
                "pool.current_memory should equal persistent leaf grad bytes"
            );

            cached_end.push(pool.cached_memory);

            // Rebuild store+graph each epoch to mimic the training loop invariant.
            let mut old = std::mem::take(&mut store.tensors);

            let mut bias = old.swap_remove(linear.bias_id);
            let mut weight = old.swap_remove(linear.weight_id);
            let mut target = old.swap_remove(target_id);
            let mut x = old.swap_remove(x_id);

            x.creator = None;
            target.creator = None;
            weight.creator = None;
            bias.creator = None;

            x.zero_grad();
            weight.zero_grad();
            bias.zero_grad();

            drop(old);

            store = TensorStore::new();
            graph = Graph::new();

            x_id = store.add(x);
            target_id = store.add(target);
            linear.weight_id = store.add(weight);
            linear.bias_id = store.add(bias);
        }

        // Fixed-shape workload: once the pool has enough cached buffers after the first epoch,
        // it should not need to grow further.
        let expected = cached_end[0];
        for (epoch, &cached) in cached_end.iter().enumerate().skip(1) {
            assert_eq!(
                cached, expected,
                "pool.cached_memory grew at epoch {} ({} -> {})",
                epoch, expected, cached
            );
        }
    }
}
