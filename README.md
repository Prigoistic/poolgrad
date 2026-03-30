# poolgrad

a tiny ML runtime in rust (with a bit of systems bite).

it builds a minimal autograd engine, a few matrix multiplication kernels, and a small memory system — just enough to explore where performance actually comes from.

this is not a framework. it’s a playground.

## what is this?

poolgrad is a single-binary runtime that implements:

* reverse-mode autograd over a dynamic graph
* multiple matmul kernels (naive, tiled, packed+simd, and an experimental mp variant)
* a simple kernel scheduler
* a gradient memory pool + lifetime-based release

core autograd, kernels, pooling, and scheduling are handwritten; rayon is used for parallel loops.

code map:

* autograd: [src/autograd/graph.rs](src/autograd/graph.rs)
* kernels + scheduler: [src/kernels/selector.rs](src/kernels/selector.rs)
* packed microkernel + tiling: [src/kernels/tiled.rs](src/kernels/tiled.rs)
* mp transform: [src/kernels/mp.rs](src/kernels/mp.rs)
* memory pool: [src/mem/pool.rs](src/mem/pool.rs)
* lifetime planner: [src/planner/lifetime.rs](src/planner/lifetime.rs)

## why?

most ML systems hide everything behind large abstractions.

this project asks:

> what actually makes neural networks fast?

* fewer multiplications?
* better memory access?
* vector instructions?
* less allocation?

## a quick look

forward builds a graph. backward walks it.

kernels are shared between forward and backward.

gradients are reused instead of reallocated (and released early when the planner says they’re dead).

```latex
\begin{tikzpicture}
\node (t) {tensor};
\node (g) [right=of t] {graph};
\node (k) [right=of g] {kernels};
\node (m) [below=of k] {memory pool};

\draw[->] (t) -- (g);
\draw[->] (g) -- (k);
\draw[->] (k) -- (m);
\end{tikzpicture}
```

## the interesting part

there are four ways to multiply matrices here:

* naive loops
* tiled (better cache use)
* packed + simd (pack panels + microkernel; uses NEON on arm64 when available, AVX2+FMA on x86_64)
* mp (single-level strassen-form block transform): fewer multiplies, more adds, more data movement

the mp idea is simple:

> can we compute the same result with fewer multiplications?

it’s strassen-like in spirit, but not recursive: it’s applied at block granularity.

## what happens

numbers below (kernel benchmark): apple m4 (arm64, 10 cores), rustc 1.93.1, `cargo run --release`, seed=0xbadc0de, `POOLGRAD_MP_MAX_SIZE=512`, warmup=2, trials=9, median reported.

observations from these runs:

* packed + simd dominates once sizes grow (e.g. 2.69x vs naive at 512)
* tiling helps, but only moderately
* mp reduces multiplies but is usually slower than packed (and can be slower than naive at mid sizes)
* no single kernel wins everywhere → scheduling matters (at 32 the scheduler picked naive, but packed tied for best)

kernel times (ms, median; square gemm):

| n | naive | tiled | packed+simd | mp |
|---:|---:|---:|---:|---:|
| 32 | 0.028 | 0.026 | 0.026 | 0.034 |
| 64 | 0.208 | 0.206 | 0.023 | 0.236 |
| 128 | 0.310 | 0.298 | 0.156 | 0.363 |
| 256 | 2.115 | 1.443 | 0.902 | 1.774 |
| 512 | 15.198 | 9.163 | 5.649 | 12.657 |

example (512 × 512):

```text
naive           15.2 ms
tiled            9.2 ms
packed+simd      5.6 ms
mp              12.7 ms
```

## memory

gradients are reused via a simple size-based pool.

training loop (pool off vs on):

| mode | allocations | reuses | peak resident bytes |
|---|---:|---:|---:|
| pool off | 30 | 0 | 28 |
| pool on | 3 | 27 | 28 |

```text
allocations: 30 -> 3
reuse:        0 -> 27
```

kernel + pool interaction (forward matmul + seeded backward; scheduler-selected kernel; one pool reused across sizes):

| n | kernel | iters | time (ms) | alloc | reuse | live peak (bytes) | resident peak (bytes) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | Naive | 10 | 0.305 | 1 | 9 | 4096 | 4096 |
| 64 | TiledPacked | 10 | 0.671 | 1 | 9 | 16384 | 20480 |
| 128 | TiledPacked | 10 | 4.931 | 1 | 9 | 65536 | 86016 |
| 256 | TiledPacked | 3 | 12.174 | 1 | 2 | 262144 | 348160 |
| 512 | TiledPacked | 1 | 118.022 | 1 | 0 | 1048576 | 1396736 |

## run it

```bash
cargo run --release
```

optional knobs:

```bash
POOLGRAD_FORCE_KERNEL=TiledPacked
POOLGRAD_PAR=1
POOLGRAD_MP_MAX_SIZE=512
POOLGRAD_BENCH_WARMUP=2
POOLGRAD_BENCH_TRIALS=9
```

## notes

* cpu only
* simd via neon (arm64) / avx2+fma (x86_64) when available
* mp is experimental — included to explore compute vs overhead tradeoffs (and has slightly higher fp error when enabled at larger sizes)
* code is intentionally small and explicit
