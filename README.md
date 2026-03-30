# PoolGrad

PoolGrad is a small Rust autograd + kernel playground that focuses on **measurable systems behavior**: memory reuse via a pool, explicit backward seeding, and CPU matmul kernels with a simple scheduler.

## Quickstart

- Run everything (training + correctness validation + kernel/step benches):

```bash
cargo run --release
```

- Faster sanity run (fewer benchmark trials):

```bash
POOLGRAD_BENCH_WARMUP=1 POOLGRAD_BENCH_TRIALS=3 cargo run --release
```

## Kernels

The scheduler exposes these `KernelType`s:

- `Naive`: straightforward triple loop (optionally parallel over rows)
- `Tiled`: scalar blocked matmul (cache-friendly baseline)
- `TiledPacked`: **packed-B panel + small microkernel** (SIMD when available)
- `TiledMP`: experimental MP/Strassen-style block transform (falls back to tiled style on edges)

### Packed + SIMD (`TiledPacked`)

`TiledPacked` improves locality by packing B into contiguous panels and then applying a small register-level microkernel:

- AArch64: NEON 4×4 microkernel (runtime selected)
- x86_64: AVX2+FMA 4×8 microkernel (runtime selected)
- Fallback: scalar 4×4 microkernel

The packed path is used for sufficiently large problems to amortize packing overhead.

## Benchmark output

`cargo run --release` prints:

- Correctness validation table (max abs error vs naive)
- Kernel benchmark table with `median` and `p95` for:
  - `Naive`, `Tiled`, `TiledPacked`, `TiledMP`
- Forward+Backward step benchmark (forward matmul + backward gradients)
- Kernel+Pool interaction experiment (scheduler choice + pool reuse metrics)

## Configuration (env)

### Parallelism

- `POOLGRAD_PAR=1|0` (default `1`)
- `POOLGRAD_PAR_MIN_ELEMS=<usize>` (default `16384`)

### MP kernel gating

- `POOLGRAD_MP_MAX_SIZE=<usize>` (default `128`)
  - If `max(m, n, p)` exceeds this, MP/Strassen-style transforms are skipped (the MP kernel falls back to tiled style, and the scheduler will not select MP).

### Bench harness

- `POOLGRAD_BENCH_WARMUP=<usize>` (default `2`)
- `POOLGRAD_BENCH_TRIALS=<usize>` (default `9`)

### Scheduler thresholds

- `POOLGRAD_SCHED_NAIVE_MAX=<usize>`
- `POOLGRAD_SCHED_TILED_MAX=<usize>`
- `POOLGRAD_SCHED_PACKED_MAX=<usize>`

Additional (shape-aware) heuristics:

- `POOLGRAD_SCHED_TINY_WORK_MAX=<usize>` (default `262144`)
- `POOLGRAD_SCHED_INNER_SMALL_MAX=<usize>` (default `32`)

The defaults are tuned to prefer `Naive` only for small sizes and move to `TiledPacked` for larger sizes.

### Logging

- `POOLGRAD_VERBOSE=1` prints per-epoch loss/output (and lifetime debug on the final epoch).

### Forcing kernel / microkernel

- `POOLGRAD_FORCE_KERNEL=Naive|Tiled|TiledPacked|TiledMP`
  - Overrides the scheduler (takes precedence over `POOLGRAD_KERNEL_PROFILE`).
- `POOLGRAD_MICROKERNEL=scalar|neon|avx2`
  - Forces the packed microkernel choice (falls back to scalar if unsupported on the current CPU).

### Kernel profiling override

- `POOLGRAD_KERNEL_PROFILE=<path>`: read a simple `size -> kernel` mapping
- `POOLGRAD_KERNEL_PROFILE_WRITE=<path>`: write the observed winners from the kernel benchmark table

## Notes on reproducibility

- The benchmark harness uses fixed RNG seeds for matrix generation.
- SIMD selection is runtime-dependent on CPU features; for stricter cross-machine comparability, either force a microkernel via `POOLGRAD_MICROKERNEL=scalar` or record machine/CPU details and keep the same build flags.
