# Performance

## Choose the draw interface

Use sequential `rand_next` calls when each draw determines the next computation.
Use `rand_fill!` for many values and reuse the destination. Use `Stateful` for
libraries that require `Random.AbstractRNG`.

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
uniforms while it is in cache, with tandem-c's vectorized polynomial `log`, `cos`, and
`sin`. On one Apple M4 task they write 4.5 GiB/s of Float64 normals and 5.2 of Float32.

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
