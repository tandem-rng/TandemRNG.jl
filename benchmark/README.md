# Reproducing the benchmarks

These runners compare public APIs through repeated measurements. The output records
the configuration, Julia and package versions, source hashes, environment, and status.
Each run requires a new output directory. A failed run retains its partial results.

## CPU setup

Clone PureRNGs beside the TandemRNG checkout:

```sh
git clone https://github.com/BJMCox/PureRNGs.jl ../PureRNGs.jl
```

Run these commands from the TandemRNG repository root:

```sh
julia --startup-file=no --project=benchmark -e 'using Pkg; Pkg.develop([PackageSpec(path="."), PackageSpec(path="../PureRNGs.jl")]); Pkg.instantiate(); Pkg.precompile()'
julia --startup-file=no --threads=8 --gcthreads=1 --project=benchmark benchmark/run.jl cpu results/cpu
```

The explicit `Pkg.develop` calls also support Julia 1.10, which does not use `[sources]`.
The benchmark environment does not change the package's dependencies.
The scripts require Julia 1.10 or newer.

Use an idle host. Stop other benchmarks and statistical jobs before measuring.
Choose the Julia thread count for the intended deployment.
The CPU report separates one-task fills from fills using each package's default
thread policy. Xoshiro and Random123 only appear in the one-task comparison.

Optional settings:

```sh
julia --startup-file=no --threads=8 --project=benchmark benchmark/run.jl cpu results/cpu-small --sizes=1024,1048576 --passes=3 --seconds=0.3
```

Each timing follows construction, allocation, first touch, and compilation.
Passes alternate generator order. The report includes minima, medians, allocation
bytes, allocation counts, and sample counts. Scalar chains cover all 18 supported
types, with at least 1,024 draws and two complete K32 groups. The report records the
draw count for each type. The reported scalar time is per draw.
Fills compare Float64, Float32, UInt32, and Bool. Throughput counts destination bytes,
including one byte per Bool. All scalar types include signed integers, complex floats,
and Char. The checksum includes every draw.

Native Tandem and its PureRNGs bridge must agree on output and end state before timing.
The bridge uses direct public calls without a benchmark-only RNG wrapper.
Tandem and each reference use their normal public implementations.
Random123 1.7.1 uses 23/52 random bits for Float32/Float64 through these sampler paths.
Tandem, PureRNGs, and Xoshiro use 24/53 bits. These are API comparisons with different
precision contracts, not claims of identical generator work.

`draws.jl` measures one-task fills of uniforms, normals, and exponentials against Xoshiro and
both Philox4x32 implementations, and the multithreaded `Stateful` fills:

```sh
julia --startup-file=no --threads=14 --project=benchmark benchmark/draws.jl 22 3
```

`benchmarks.jl` also exposes `SUITE` for BenchmarkTools and
`compare_cpu(io; sizes, passes, seconds, samples, seed, threaded)` for Julia callers.

## CUDA setup

The GPU runner requires CUDA 6, an NVIDIA GPU, and `nvidia-smi` on PATH.
Use the separate environment:

```sh
julia --startup-file=no --project=benchmark/cuda -e 'using Pkg; Pkg.develop([PackageSpec(path="."), PackageSpec(path="../PureRNGs.jl")]); Pkg.instantiate(); Pkg.precompile()'
julia --startup-file=no --threads=8 --gcthreads=1 --project=benchmark/cuda benchmark/run.jl gpu results/cuda --device=0
```

The runner checks device use before and after each group. It refuses another compute
process, and requires zero utilization before timing. Final utilization can include
the runner's own work. It disables scalar GPU indexing and checks Tandem's
native and bridge outputs against CPU results before timing.

The default fills use 2^20 and 2^27 elements. The latter needs 1 GiB for a Float64
destination. Reduce `--sizes` for smaller devices. The script releases each destination
before allocating the next case. Each case warms for 0.5 seconds and records the
minimum of 30 CUDA event timings. Three passes alternate generator order.
Host allocation bytes include the synchronized launch. They exclude destination allocation.

The public scalar report measures 1,024 draws per thread from bound RNG states.
Its throughput counts generated Float64 bytes, not global-memory writes.
The core report measures the library cores separately. Random123 has no GPU array-fill
API here, so it appears only in that core comparison. CURAND's library API has no Bool case.

## First-use cost

Run this after package precompilation:

```sh
julia --startup-file=no --project=benchmark benchmark/run.jl latency results/first-use --passes=3
```

Each engine starts in a fresh process with two Julia threads and one GC thread.
The report separates package load, construction, scalar draws, and small fills.
It records wall time, allocations, and compilation time where Julia supplies it.
Package caches remain warm. This measures first use after installation, not dependency
downloads, package precompilation, or operating-system cache misses.

For Tandem's broader API coverage, `latency.jl` remains available as a separate workload.

## Comparing results

Keep the full output directory and runner sources. Environment manifests may contain
local checkout paths. Replace those paths when reproducing on another host.
Use source hashes to check the package and runner revisions.
Compare repeated ranges on the same idle host. A single minimum is not a stable speed ratio.
Measure on an idle machine and record CPU policy and GPU clocks when publishing numbers.
Do not combine core throughput with array-fill throughput in one speed claim.
