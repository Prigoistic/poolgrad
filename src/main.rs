mod autograd;
mod tensor;
mod nn;
mod plannar;
mod mem;
mod kernels;

use std::time::Instant;

use autograd::graph::Graph;
use tensor::store::TensorStore;
use tensor::tensor::Tensor;
use nn::linear::Linear;
use nn::loss::mse;
use mem::pool::MemoryPool;
use plannar::lifetime::compute_lifetimes;
use kernels::naive::matmul_naive;
use kernels::tiled::matmul_tiled;
use kernels::selector::select_kernel;
use kernels::selector::{KernelType, matmul as matmul_by_kernel};

struct Config {
    use_pool: bool,
    verbose: bool,
}

fn run_training(config: &Config) -> MemoryPool {
    let mut store = TensorStore::new();
    let mut graph = Graph::new();
    let mut pool = MemoryPool::new();
    pool.enabled = config.use_pool;
    let x_id = store.add(Tensor::new(vec![1.0, 2.0], vec![1, 2], true));
    let target_id = store.add(Tensor::new(vec![3.0, 3.0, 3.0], vec![1, 3], false));

    let linear = Linear::new(2, 3, &mut store);

    for epoch in 0..10 {
        // forward
        let out_id = linear.forward(x_id, &mut store, &mut graph, &mut pool);
        let loss_id = mse(out_id, target_id, &mut store, &mut graph, &mut pool);

        // backward
        graph.backward(&mut store, loss_id);

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

        // planner + pool: free intermediate grad buffers at their death step
        let lifetimes = compute_lifetimes(&graph);

        if config.verbose && epoch == 9 {
            for (id, lt) in lifetimes.iter() {
                println!("Tensor {}: birth={}, death={}", id, lt.birth, lt.death);
            }
        }

        // Bucket tensors by death step to avoid scanning all lifetimes each step.
        let mut death_buckets: Vec<Vec<usize>> = vec![Vec::new(); graph.nodes.len()];
        for (tensor_id, lifetime) in lifetimes.iter() {
            if lifetime.death < death_buckets.len() {
                let tensor_id = *tensor_id;
                // Only release intermediate tensors (creator set). Persistent tensors like params/inputs keep grads.
                if store.get(tensor_id).creator.is_some() {
                    death_buckets[lifetime.death].push(tensor_id);
                }
            }
        }

        for (step, _node) in graph.nodes.iter().enumerate() {
            for &tensor_id in &death_buckets[step] {
                let tensor = store.get_mut(tensor_id);
                if !tensor.grad.is_empty() {
                    pool.release(std::mem::take(&mut tensor.grad));
                }
            }
        }

        // reset graph + grads (IMPORTANT)
        graph.nodes.clear();
        for t in &mut store.tensors {
            t.zero_grad();
        }
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

    let peak_bytes = pool.peak_memory;

    println!("=== {} ===", label);
    println!(
        "Mode | Allocations | Reuses | ReuseRate | Peak (bytes)\n{} | {} | {} | {:.2}% | {}",
        label,
        pool.allocations,
        pool.reuses,
        reuse_rate,
        peak_bytes
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

    println!("\nCorrectness validation (seed={:#x}, tol=1e-4)", seed);
    println!("Size | Kernel  | MaxAbsErr");
    println!("-----|---------|----------");

    for &size in &sizes {
        let a = Tensor::new(random_matrix(&mut rng, size, size), vec![size, size], false);
        let b = Tensor::new(random_matrix(&mut rng, size, size), vec![size, size], false);

        let reference = matmul_naive(&a, &b);

        for (kernel, name) in [
            (KernelType::Tiled, "Tiled"),
            (KernelType::TiledMP, "TiledMP"),
        ] {
            let out = matmul_by_kernel(&a, &b, kernel);
            let err = max_abs_diff(&reference.data, &out.data);
            println!("{:>4} | {:<7} | {:>8.2e}", size, name, err);
            assert!(
                err < 1e-4,
                "kernel {:?} failed validation at size {}: max_abs_err={} (tol=1e-4)",
                kernel,
                size,
                err
            );
        }
    }
}

fn time_ms<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() * 1000.0
}

fn run_kernel_benchmark() {
    let seed = 0xBADC0DE_u64;
    let mut rng = XorShift64::new(seed);
    let sizes = [16usize, 32, 64, 96, 128, 256];

    println!("\nKernel benchmark (seed={:#x})", seed);
    println!("Size | Naive(ms) | Tiled(ms) | TiledMP(ms) | Winner  | Scheduler");
    println!("-----|----------|----------|------------|---------|----------");

    for &size in &sizes {
        let a = Tensor::new(random_matrix(&mut rng, size, size), vec![size, size], false);
        let b = Tensor::new(random_matrix(&mut rng, size, size), vec![size, size], false);

        let reference = matmul_naive(&a, &b);

        let naive_ms = time_ms(|| {
            let _ = matmul_naive(&a, &b);
        }, 1);

        let tiled_out = matmul_tiled(&a, &b, 16);
        let tiled_err = max_abs_diff(&reference.data, &tiled_out.data);
        assert!(tiled_err < 1e-4, "Tiled failed validation at size {}: err={}", size, tiled_err);
        let tiled_ms = time_ms(|| {
            let _ = matmul_tiled(&a, &b, 16);
        }, 1);

        let tiled_mp_out = kernels::tiled_mp::matmul_tiled_mp(&a, &b, 16);
        let tiled_mp_err = max_abs_diff(&reference.data, &tiled_mp_out.data);
        assert!(
            tiled_mp_err < 1e-4,
            "TiledMP failed validation at size {}: err={}",
            size,
            tiled_mp_err
        );
        let tiled_mp_ms = time_ms(|| {
            let _ = kernels::tiled_mp::matmul_tiled_mp(&a, &b, 16);
        }, 1);

        let (winner, _best_ms) = if tiled_ms <= naive_ms && tiled_ms <= tiled_mp_ms {
            ("Tiled", tiled_ms)
        } else if tiled_mp_ms <= naive_ms && tiled_mp_ms <= tiled_ms {
            ("TiledMP", tiled_mp_ms)
        } else {
            ("Naive", naive_ms)
        };

        let scheduled = select_kernel(size);

        println!(
            "{:>4} | {:>8.3} | {:>8.3} | {:>10.3} | {:<7} | {:?}",
            size, naive_ms, tiled_ms, tiled_mp_ms, winner, scheduled
        );
    }
}

fn main() {
    // Ensure reproducible, end-to-end comparable metrics across runs.
    MemoryPool::reset_global_metrics();

    let verbose = std::env::var("POOLGRAD_VERBOSE").ok().as_deref() == Some("1");
    let baseline = Config { use_pool: false, verbose };
    let pooled = Config { use_pool: true, verbose };

    let baseline_pool = run_training(&baseline);
    let pooled_pool = run_training(&pooled);

    print_metrics("Baseline (pool OFF)", &baseline_pool);
    print_metrics("Pooled (pool ON)", &pooled_pool);

    validate_kernels();
    run_kernel_benchmark();

    let exp_peak = run_kernel_pool_interaction_experiment();

    let global_peak = MemoryPool::global_peak_memory_bytes();
    println!("\nGlobal peak memory (bytes): {}", global_peak);
    println!("Interaction experiment peak (bytes): {}", exp_peak);
    assert_eq!(
        global_peak, exp_peak,
        "global peak ({}) != experiment peak ({})",
        global_peak, exp_peak
    );
}

fn run_kernel_pool_interaction_experiment() -> usize {
    use kernels::selector::KernelType;
    use tensor::tensor::matmul_scheduled_with_pool;

    let sizes = [16usize, 32, 64, 128, 256];
    let mut pool = MemoryPool::new();
    pool.enabled = true;

    println!("\nKernel+Pool interaction experiment (scheduler drives kernel, pool drives grad reuse)");
    println!("Size | Kernel | Iters | Time (ms) | Alloc | Reuse | Peak (bytes)");
    println!("-----|--------|-------|-----------|-------|-------|-------------");

    let mut exp_peak = 0usize;

    for &size in &sizes {
        let iters = if size <= 128 { 10usize } else { 3usize };
        let kernel: KernelType = select_kernel(size);

        let alloc0 = pool.allocations;
        let reuse0 = pool.reuses;

        let start = Instant::now();
        for _ in 0..iters {
            let mut store = TensorStore::new();
            let mut graph = Graph::new();

            let a_id = store.add(Tensor::new(vec![1.0; size * size], vec![size, size], true));
            let b_id = store.add(Tensor::new(vec![1.0; size * size], vec![size, size], true));

            let out_id = matmul_scheduled_with_pool(a_id, b_id, &mut store, &mut graph, &mut pool);
            graph.backward(&mut store, out_id);

            // planner + pool: free intermediate grad buffers at their death step
            let lifetimes = compute_lifetimes(&graph);
            let mut death_buckets: Vec<Vec<usize>> = vec![Vec::new(); graph.nodes.len()];
            for (tensor_id, lifetime) in lifetimes.iter() {
                if lifetime.death < death_buckets.len() {
                    let tensor_id = *tensor_id;
                    if store.get(tensor_id).creator.is_some() {
                        death_buckets[lifetime.death].push(tensor_id);
                    }
                }
            }

            for (step, _node) in graph.nodes.iter().enumerate() {
                for &tensor_id in &death_buckets[step] {
                    let tensor = store.get_mut(tensor_id);
                    if !tensor.grad.is_empty() {
                        pool.release(std::mem::take(&mut tensor.grad));
                    }
                }
            }
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let alloc = pool.allocations - alloc0;
        let reuse = pool.reuses - reuse0;

        println!(
            "{:>4} | {:>6?} | {:>5} | {:>9.3} | {:>5} | {:>5} | {}",
            size, kernel, iters, elapsed_ms, alloc, reuse, pool.peak_memory
        );

        exp_peak = exp_peak.max(pool.peak_memory);
    }

    exp_peak
}