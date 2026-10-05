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

Apple M4, Julia 1.13.1, 2026-10-05. One task, fills of 2^22 elements, GiB/s written, best of
three BenchmarkTools passes with alternating generator order, `benchmark/draws.jl`. Tandem
calls the immutable fills with `nthreads = 1`, PureRNGs its fills with `threaded = false`, and
Xoshiro and Random123 the `Random` API.

| one task | Tandem | Xoshiro | PureRNGs Philox4x32 | Random123 Philox4x32 |
|---|---|---|---|---|
| `rand!` Float64 | 17.7 | 21.4 | 4.80 | 1.83 |
| `rand!` Float32 | 17.5 | 20.2 | 5.09 | 1.74 |
| `rand!` UInt32 | 21.7 | 27.0 | 4.71 | 1.79 |
| `randn!` Float64 | 6.34 | 7.28 | 1.43 | 1.51 |
| `randn!` Float32 | 5.50 | 1.28 | 1.09 | 0.75 |
| `randexp!` Float64 | 6.02 | 6.45 | 2.28 | 1.31 |
| `randexp!` Float32 | 6.82 | 1.16 | 2.94 | 0.65 |

In the same window tandem-c reaches 16.6, 16.6 and 19.7 GiB/s for the Float64, Float32 and
UInt32 fills, 7.7 and 5.6 for the normals and 6.2 and 6.9 for the exponentials. Xoshiro's
normals and exponentials are Julia's ziggurats, and PureRNGs' are its own samplers.

The same fills through `Stateful` on 14 threads:

| 14 tasks | `rand!` | `randn!` | `randexp!` |
|---|---|---|---|
| Float64 | 108 | 46.5 | 43.1 |
| Float32 | 109 | 37.2 | 42.5 |
| UInt32 | 127 | | |

Float64 normals are the ziggurat: a table pass over each group of UInt64 draws, with the 0.43 %
of draws that miss resolved on their fallback one at a time. Float32 normals and the
exponentials run tandem-c's polynomials, vectorized two doubles or four floats wide with four
interleaved iterations. Every float fill converts the words in the row loop before the store.

Tandem on AMD EPYC 7702P (AVX2), Julia 1.13, 2026-09-26. One task, 2^20 elements,
three BenchmarkTools passes with alternating generator order. Scalar chains include
at least two complete reseeding periods. Every case below allocates zero bytes.

| generator | chain, ns/draw | Float64 fill, GiB/s | Float32 fill, GiB/s | UInt32 fill, GiB/s |
|---|---|---|---|---|
| Tandem native | 2.49–2.50 | 7.65–7.66 | 10.70–10.72 | 14.75–14.76 |
| Tandem bridge | 2.48–2.50 | 7.65–7.66 | 10.70–10.71 | 14.75–14.76 |
| PureRNGs Philox4x32 | 7.72–7.77 | 1.27–1.27 | 1.58–1.59 | 1.53–1.53 |
| PureRNGs Philox4x64 | 6.21–6.23 | 1.28–1.30 | 1.77–1.78 | 1.47–1.48 |
| Random123 Philox4x32 | 10.50–10.52 | 0.77–0.77 | 0.68–0.68 | 0.72–0.72 |
| Random123 Philox4x64 | 7.10–7.13 | 1.40–1.40 | 0.71–0.71 | 0.76–0.78 |
| Xoshiro | 1.29–1.29 | 6.41–6.41 | 14.23–14.23 | 16.61–16.62 |

Random123 1.7.1 supplies 23/52 random bits for these Float32/Float64 APIs; the other
generators supply 24/53. This table predates the Float32 fill through the full-group loop.

### GPU

NVIDIA A100 40 GB core measurements (2026-09-21), idle GPU, three passes. Compilation precedes a 0.5-second
warm-up; each figure uses the minimum of 30 CUDA event timings.

| | Tandem K=32 | PureRNGs Philox4x32 | Random123 Philox4x32 |
|---|---|---|---|
| core draws, one chain/thread, GiB/s generated | 4657–4682 | 2891–2895 | 2861–2901 |

The core comparison calls both libraries' actual Philox functions in identical kernels,
folding all generated words into one stored checksum per chunk. Tandem includes seeding.
Its core throughput is about 1.6× either reference.

A100 fills (2026-10-02), idle GPU, three passes, minimum of 30 CUDA event timings after a 0.5-second warm-up. The fill stages the
rows of each workgroup in shared memory so a warp writes 512 contiguous bytes.

| elements | Float32 fill, GiB/s | UInt32 fill, GiB/s | Float64 fill, GiB/s |
|---|---|---|---|
| Tandem K=32, 2^28 | 1374–1375 | 1372–1383 | 1402–1404 |
| Tandem K=32, 2^27 | 1288–1302 | 1144–1299 | 1347–1351 |
| PureRNGs Philox4x32, 2^27 | 1313–1368 | 1197–1295 | 1201–1206 |
| CUDA.jl native, 2^27 | 571–588 | 1076–1107 | 1077–1084 |

At 2^27 the fixed launch and clock-ramp cost shows, which is why the two sizes differ.
Chained scalar Float64 calls through the public A100 API reach about 597 GiB/s. These
measure different work.

Apple M4 Pro through Metal.jl, minimum of seven after a 0.5 s warm-up: UInt32
fill 164 GiB/s at 2^26 elements (126 at 2^24), UInt64 163 at 2^25, Float32 155 at 2^26.
A constant-store kernel reaches 720 GiB/s on the same GPU, so the Apple fill is bound by
integer throughput, not by memory. `benchmark/metal/` holds the environment.

## Reproduce measurements

The [benchmark guide](https://github.com/tandem-rng/TandemRNG.jl/blob/main/benchmark/README.md)
provides separate CPU, CUDA, and first-use runners. It compares Tandem's native
interface and PureRNGs bridge with PureRNGs, Random123, and available platform generators.
It documents setup, supported types, timing protocols, and API precision differences.

Use an idle host and repeated passes. Separate compilation, allocation, and first
touch from sustained draw time. Report generated-byte throughput separately from
array-write throughput. Keep runner sources, manifests, source hashes, and full
output directories with published results.

The first-use runner starts fresh processes after package precompilation.
Its load measurements exclude process startup. Later calls can reuse code compiled
by earlier calls, so they are not independent cold measurements.
