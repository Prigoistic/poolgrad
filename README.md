# PoolGrad

PoolGrad is a memory-aware ML runtime in Rust designed to study compute–memory tradeoffs in CPU matmul kernels and reverse-mode autograd.

## 2. What It Is

Single-binary research playground with:

- `Tensor` + `TensorStore` and a minimal reverse-mode tape (`Graph`)
- a shape-aware matmul kernel selector
- a size-classed gradient `MemoryPool`
- a lifetime planner that releases intermediate grads early

## 3. System Overview

Ops append nodes to a `Graph`; backward walks the tape in reverse and dispatches matmul gradients through the same kernel selector. Intermediate gradient buffers are returned to the pool when their last use passes.

```latex
\begin{tikzpicture}
\node (tensor) {Tensor + Store};
\node (graph) [right=of tensor] {Graph (tape)};
\node (sched) [right=of graph] {Kernel Selector};
\node (kernel) [right=of sched] {MatMul Kernels};
\node (mem) [below=of kernel] {MemoryPool};
\node (plan) [below=of graph] {Lifetime Planner};

\draw[->] (tensor) -- (graph);
\draw[->] (graph) -- (sched);
\draw[->] (sched) -- (kernel);
\draw[->] (kernel) -- (mem);
\draw[->] (graph) -- (plan);
\draw[->] (plan) -- (mem);
\end{tikzpicture}
```

## 4. Key Components

- Autograd: reverse-mode tape over `Operation::{Add, Mul, MatMul, ReLU, MSE}` ([src/autograd/graph.rs](src/autograd/graph.rs)).
- Kernels: `KernelType::{Naive, Tiled, TiledPacked, TiledMP}` with optional Rayon parallelism ([src/kernels](src/kernels)).
- Memory Pool: size-keyed free lists of `Vec<f32>` with alloc/reuse + peak tracking ([src/mem/pool.rs](src/mem/pool.rs)).
- Scheduler: shape-aware selection with env overrides (`POOLGRAD_FORCE_KERNEL`, `POOLGRAD_SCHED_*`) ([src/kernels/selector.rs](src/kernels/selector.rs)).

## 5. Experiments

Bench environment:

- Apple M4 (arm64, 10 cores), macOS; rustc 1.93.1
- `cargo run --release`
- `POOLGRAD_BENCH_WARMUP=2`, `POOLGRAD_BENCH_TRIALS=9`, `POOLGRAD_MP_MAX_SIZE=512`

Kernel performance (square GEMM, median over trials; speedup vs Naive):

| Size | Kernel | Time (ms, median) | Speedup vs Naive |
|---:|---|---:|---:|
| 32 | Naive | 0.028 | 1.00 |
| 32 | Tiled | 0.026 | 1.08 |
| 32 | Packed+SIMD | 0.026 | 1.08 |
| 32 | MP | 0.034 | 0.82 |
| 64 | Naive | 0.208 | 1.00 |
| 64 | Tiled | 0.206 | 1.01 |
| 64 | Packed+SIMD | 0.023 | 9.04 |
| 64 | MP | 0.236 | 0.88 |
| 128 | Naive | 0.310 | 1.00 |
| 128 | Tiled | 0.298 | 1.04 |
| 128 | Packed+SIMD | 0.156 | 1.99 |
| 128 | MP | 0.363 | 0.85 |
| 256 | Naive | 2.115 | 1.00 |
| 256 | Tiled | 1.443 | 1.47 |
| 256 | Packed+SIMD | 0.902 | 2.34 |
| 256 | MP | 1.774 | 1.19 |
| 512 | Naive | 15.198 | 1.00 |
| 512 | Tiled | 9.163 | 1.66 |
| 512 | Packed+SIMD | 5.649 | 2.69 |
| 512 | MP | 12.657 | 1.20 |

Memory pooling (training loop, pool OFF vs ON):

| Mode | Alloc | Reuse | Peak (bytes) |
|---|---:|---:|---:|
| Baseline (pool OFF) | 30 | 0 | 28 |
| Pooled (pool ON) | 3 | 27 | 28 |

Allocation reduction vs baseline: 90.00%

Combined: kernel + pool interaction (forward matmul + seeded backward; scheduler-selected kernel; one pool reused across sizes):

| Size | Kernel | Iters | Time (ms) | Alloc | Reuse | LivePeak (bytes) | ResidentPeak (bytes) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | Naive | 10 | 0.305 | 1 | 9 | 4096 | 4096 |
| 64 | TiledPacked | 10 | 0.671 | 1 | 9 | 16384 | 20480 |
| 128 | TiledPacked | 10 | 4.931 | 1 | 9 | 65536 | 86016 |
| 256 | TiledPacked | 3 | 12.174 | 1 | 2 | 262144 | 348160 |
| 512 | TiledPacked | 1 | 118.022 | 1 | 0 | 1048576 | 1396736 |

## 6. Key Insights

- Packed+SIMD dominates once sizes are non-trivial: 9.04x vs Naive at 64, and 2.69x at 512.
- Tiled is close to Naive at 64/128 but scales better at 256/512 (1.47–1.66x vs Naive).
- MP/Strassen-form blocks are not a win here: slower than Packed+SIMD at every tested size; only modestly faster than Naive at 256/512.
- Pooling removes most per-step grad allocations on fixed-shape workloads (30 → 3 allocs, 27 reuses here).
- The scheduler is intentionally heuristic: at size 32, it selected Naive while Packed+SIMD was the measured winner; thresholds are configurable.

## 7. How to Run

```bash
cargo run --release
```

Useful knobs (env):

- Select kernel: `POOLGRAD_FORCE_KERNEL=Naive|Tiled|TiledPacked|TiledMP`
- Parallelism: `POOLGRAD_PAR=0|1`, `POOLGRAD_PAR_MIN_ELEMS=<usize>`
- MP gating: `POOLGRAD_MP_MAX_SIZE=<usize>`
- Bench stability: `POOLGRAD_BENCH_WARMUP=<usize>`, `POOLGRAD_BENCH_TRIALS=<usize>`
- Packed microkernel: `POOLGRAD_MICROKERNEL=scalar|neon|avx2`

## 8. Notes

- CPU-only; packed microkernel uses runtime feature detection (NEON on AArch64, AVX2+FMA on x86_64 when available).
- `TiledMP` implements a single-level Strassen-form transform over 2×2 sub-blocks; it trades extra adds for fewer multiplies and slightly higher numerical error when enabled for larger sizes.
