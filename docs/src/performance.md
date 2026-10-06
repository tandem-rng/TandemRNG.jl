# Performance

## Choose the draw interface

Use sequential `rand_next` calls when each draw determines the next computation.
Use `rand_fill!` for many values and reuse the destination. Use `Stateful` for
libraries that require `Random.AbstractRNG`. `Stateful` caches the current row for
sequential draws through the `Random` API.

`rand_at` reconstructs each component from its chunk seed and up to `K` recurrence
steps. Use it for addressed access. Sequential draws reuse cached state.

CPU fills distribute groups across Julia threads. Small arrays can be slower with
many tasks. Pass `nthreads = 1` when an outer loop already uses threads:

```@example performance
using TandemRNG
rng = Tandem8x32(42)
buffer = Vector{Float32}(undef, 1024)
rng = rand_fill!(rng, buffer; nthreads = 1)
rngposition(rng)
```

`normal_fill!` and `exponential_fill!` take the same keyword. They map each group of
draws while it is in cache, the Float64 normals by the ziggurat's tables and the others with
tandem-c's vectorized polynomial `log`, `cos`, and `sin`. See
[Measurements](@ref performance-measurements) for their speed.

For GPU work, allocate on the bound backend and synchronize before measuring completion.
Host scalar calls on a GPU-bound generator still run on the host.

## Precompilation

PrecompileTools runs automatically during package precompilation. CPU workloads
cover all 17 chunk lengths, all 18 draw types, fills, derivations, and `Stateful`.
The default chunk length also covers matrices, views, allocating draws, and bulk
normal and exponential draws. This coverage increases precompile time and cache size.

Optional backend extensions precompile host methods when the backend,
KernelAbstractions, and GPUArraysCore are installed. Metal also needs GPUArrays.
Workloads cover vectors, matrices, supported draw types, and K1/K32/K64.

Backend precompilation requires Julia 1.10.11 or later in 1.10, or Julia 1.11.2
or later. Earlier supported Julia versions retain runtime GPU support.
Device compilation also requires Julia 1.12 or later and a working device.
Coverage and forced bounds-checking builds skip device workloads. Metal caches
host methods during downstream package precompilation without initializing a device.

Persistent device-code reuse depends on the backend, compiler, and target.
First use on another device can still compile. Reactant precompiles conversion
and tracing entry points. The first `Reactant.@compile` still builds the executable
for its concrete backend and shape.

No GPU is required to load or precompile TandemRNG. Use `benchmark/latency.jl` in fresh
Julia processes to measure first-use latency.

## [Measurements](@id performance-measurements)

### CPU

Apple M4 Pro, Julia 1.13.1, 2026-10-06, all CPU and Metal figures from one session. One
task, fills of 2^22 elements, GiB/s written, best of three BenchmarkTools passes with
alternating generator order, `benchmark/draws.jl`. Tandem calls the immutable fills with
`nthreads = 1`, and Xoshiro and Random123 the `Random` API.

| one task | Tandem | Xoshiro | Random123 Philox4x32 |
|---|---|---|---|
| `rand!` Float64 | 17.2 | 19.8 | 1.81 |
| `rand!` Float32 | 16.8 | 19.6 | 1.71 |
| `rand!` UInt32 | 19.6 | 26.3 | 1.77 |
| `randn!` Float64 | 7.74 | 7.19 | 1.49 |
| `randn!` Float32 | 5.54 | 1.25 | 0.74 |
| `randexp!` Float64 | 6.16 | 6.35 | 1.29 |
| `randexp!` Float32 | 6.78 | 1.13 | 0.65 |

In the same session tandem-c reaches 16.6, 16.5 and 19.1 GiB/s for the Float64, Float32 and
UInt32 fills, 7.7 and 5.5 for the normals and 6.1 and 6.7 for the exponentials. A UInt32 row
of 128 bytes takes 81 vector instructions, so the uniform fills run at the vector issue rate
of the core. Xoshiro's normals and exponentials are Julia's ziggurats.

The same fills on 14 threads. Tandem fills through `Stateful`. Xoshiro and Random123 have
no threaded fill, so each task fills its own contiguous chunk with its own generator.

| 14 tasks | Tandem `Stateful` | Xoshiro | Random123 Philox4x32 |
|---|---|---|---|
| `rand!` Float64 | 111 | 117 | 11.2 |
| `rand!` Float32 | 108 | 126 | 10.3 |
| `rand!` UInt32 | 126 | 150 | 11.3 |
| `randn!` Float64 | 49.4 | 47.9 | 9.22 |
| `randn!` Float32 | 36.7 | 9.76 | 4.76 |
| `randexp!` Float64 | 41.4 | 41.9 | 8.09 |
| `randexp!` Float32 | 43.7 | 10.8 | 4.18 |

Scalar chains of 1024 Float64 draws, which span two complete K = 32 groups and so include
the reseeding, one task, three passes with alternating generator order, GiB/s at 8 bytes per
draw, `compare_cpu` in `benchmark/benchmarks.jl`. Tandem chains `rand_next`, and the
references draw through the `Random` sampler.

| generator | chain, GiB/s |
|---|---|
| Tandem | 7.60–7.60 |
| Random123 Philox4x32 | 1.83–1.83 |
| Random123 Philox4x64 | 2.99–3.05 |
| Xoshiro | 10.2–10.4 |

tandem-c's own Float64 chain benchmark reaches 5.2 GiB/s in the same session. Each Tandem draw that ends
inside its block reads the held block and changes no state. Every second Float64 draw
crosses a block and rotates the row state by one lane.

Float64 normals are the ziggurat: a table pass over each group of UInt64 draws, three draws
per test. The 0.43 % of draws that miss queue across groups and resolve eight at a time on
fallbacks seeded eight wide, as in tandem-c. Float32 normals and the
exponentials run tandem-c's polynomials, vectorized two doubles or four floats wide with four
interleaved iterations. Every float fill converts the words in the row loop before the store.

Tandem on AMD EPYC 7702P (AVX2), Julia 1.13, 2026-09-26. One task, 2^20 elements,
three BenchmarkTools passes with alternating generator order. Every case below allocates
zero bytes.

| generator | Float64 fill, GiB/s | Float32 fill, GiB/s | UInt32 fill, GiB/s |
|---|---|---|---|
| Tandem | 7.65–7.66 | 10.70–10.72 | 14.75–14.76 |
| Xoshiro | 6.41–6.41 | 14.23–14.23 | 16.61–16.62 |
| Random123 Philox4x64 | 1.40–1.40 | 0.71–0.71 | 0.76–0.78 |
| Random123 Philox4x32 | 0.77–0.77 | 0.68–0.68 | 0.72–0.72 |

Random123 1.7.1 supplies 23/52 random bits for these Float32/Float64 APIs; the other
generators supply 24/53. This table predates the Float32 fill through the full-group loop.

### GPU

NVIDIA A100 40 GB core measurements (2026-09-21), idle GPU, three passes. Compilation precedes a 0.5-second
warm-up; each figure uses the minimum of 30 CUDA event timings.

| | Tandem K=32 | Random123 Philox4x32 |
|---|---|---|
| core draws, one chain/thread, GiB/s generated | 4657–4682 | 2861–2901 |

The core comparison calls Random123's actual Philox function in an identical kernel,
folding all generated words into one stored checksum per chunk. Tandem includes seeding.
Its core throughput is about 1.6× Random123's.

A100 fills (2026-10-02), idle GPU, three passes, minimum of 30 CUDA event timings after a 0.5-second warm-up. The fill stages the
rows of each workgroup in shared memory so a warp writes 512 contiguous bytes.

| elements | Float32 fill, GiB/s | UInt32 fill, GiB/s | Float64 fill, GiB/s |
|---|---|---|---|
| Tandem K=32, 2^28 | 1374–1375 | 1372–1383 | 1402–1404 |
| Tandem K=32, 2^27 | 1288–1302 | 1144–1299 | 1347–1351 |
| CUDA.jl native, 2^27 | 571–588 | 1076–1107 | 1077–1084 |

At 2^27 the fixed launch and clock-ramp cost shows, which is why the two sizes differ.
Chained scalar Float64 calls through the public A100 API reach about 597 GiB/s. These
measure different work.

Apple M4 Pro GPU through Metal.jl 1.11.1, in the CPU session above. Each figure is the
minimum of seven synchronized fills after a 0.5 s warm-up, best of three passes with
alternating generator order, `benchmark/metal/fills.jl`. Metal.jl's `rand!` draws these
types from Metal Performance Shaders' Philox.

| elements | Tandem | Metal.jl `rand!` |
|---|---|---|
| UInt32, 2^24 | 74.6 | 127 |
| UInt32, 2^26 | 164 | 184 |
| UInt64, 2^25 | 165 | 175 |
| Float32, 2^26 | 153 | 183 |

A synchronized Tandem fill of 1024 elements takes about 0.2 ms, twice Metal.jl's, which
dominates at 2^24. A constant-store kernel reaches 720 GiB/s on the same GPU, so the Apple
fill is bound by integer throughput, not by memory.

## Reproduce measurements

The [benchmark guide](https://github.com/tandem-rng/TandemRNG.jl/blob/main/benchmark/README.md)
provides separate CPU, CUDA, Metal, and first-use runners. It compares Tandem with
third-party generators only: Xoshiro, Random123, CUDA.jl, CURAND, and Metal.jl.
It documents setup, supported types, timing protocols, and API precision differences.

Use an idle host and repeated passes. Separate compilation, allocation, and first
touch from sustained draw time. Report generated-byte throughput separately from
array-write throughput. Keep runner sources, manifests, source hashes, and full
output directories with published results.

The first-use runner starts fresh processes after package precompilation.
Its load measurements exclude process startup. Later calls can reuse code compiled
by earlier calls, so they are not independent cold measurements.
