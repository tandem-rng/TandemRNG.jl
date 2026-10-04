<p align="center"><img src="docs/src/assets/lockup.png" width="560" alt="tandem rng .jl"></p>

# TandemRNG

[![Build Status](https://github.com/tandem-rng/TandemRNG.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tandem-rng/TandemRNG.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/tandem-rng/TandemRNG.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tandem-rng/TandemRNG.jl)
[![Documentation](https://img.shields.io/badge/docs-guide-blue.svg)](https://bjmcox.github.io/TandemRNG.jl/)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10%2B-9558B2?logo=julia)](https://julialang.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Tandem is a random number generator built to be fast on CPUs and GPUs alike: **T**wo-half **A**symmetric
**N**onlinear **D**uplex with **E**volving **M**ultipliers.

**TandemRNG is not a cryptographic PRNG. Do not use it for cryptography or
security-sensitive applications.**

- Eight 32-bit words, using 32x32→64 multiplication, xor, rotate, and add.
- A hidden half runs a bijective clock and receives feedback from the new exposed half.
  The exposed half is a Philox-shaped Feistel whose multipliers and masks come from the hidden half.
- The stream is cut into chunks of `K` steps. A keyed seeding function starts every chunk, so
  splitting and forking cost one seeding per pair of children. Random access adds at most
  `K` steps after seeding.
- Chunks come in groups of eight, and the stream is a sequence of 128-byte rows: the eight
  16-byte blocks of a group at one step. On a GPU each work-item owns one chunk and eight
  adjacent work-items write one contiguous row per step, direct 16-byte stores, no shared
  memory. On a CPU the eight chunks step together, with one eight-lane vector per state word.
- A bit-aligned stream law. Scalar components stay inside 128-bit blocks. Complex draws
  compose two real draws. Fills and scalar loops agree on every supported backend.
- All 18 PureRNGs result types: Bool, 8–128-bit integers, Float16/32/64, their complex
  types, and Char. Device fills exclude 128-bit integers. Metal also excludes Float64
  and ComplexF64. Reactant supports Bool, 8–64-bit integers, and Float16/32/64.
- The transport form of a generator is its 128-bit key plus a 64-bit bit position.

[SPEC.md](SPEC.md) defines the algorithms, stream order, draw mappings, and test vectors.

## Status

The default variant is `Tandem8x32-K32`. The package supports Julia 1.10 and
later, CPU arrays, and optional CUDA, AMDGPU, Metal, Reactant, and PureRNGs integrations.
AMDGPU hardware validation remains open. Version 0.1.0 has no package release yet,
and the package is not registered.

The [user documentation](https://bjmcox.github.io/TandemRNG.jl/) covers draws,
parallel streams, device binding, integrations, and reproducibility.
The [validation evidence](https://github.com/tandem-rng/TandemRNG.jl/releases/tag/statistical-evidence-2026-09-27)
contains logs, case matrices, protocols, input hashes, frozen reproduction scripts,
and flagged results from PractRand, BigCrush, HWD, gjrand, and reduced-round tests.

- [Main validation records](https://github.com/tandem-rng/TandemRNG.jl/releases/download/statistical-evidence-2026-09-27/tandemrng-validation-records-20260926.7z)
- [Reduced-round margin records](https://github.com/tandem-rng/TandemRNG.jl/releases/download/statistical-evidence-2026-09-27/tandemrng-validation-margin-20260927.7z)
- [Validation audit](https://github.com/tandem-rng/TandemRNG.jl/releases/download/statistical-evidence-2026-09-27/tandemrng-validation-audit-20260926.tar.zst)

The three archives total 5.42 MiB. Verify downloads against
[SHA256SUMS](https://github.com/tandem-rng/TandemRNG.jl/releases/download/statistical-evidence-2026-09-27/SHA256SUMS).

CI runs on pushes to `main`, version tags, and pull requests. Coverage reports the CPU package tests.
GPU validation and optional integration suites run separately.

## Use

```julia
using TandemRNG

rng = Tandem8x32(42)                    # chunks of 32 steps, seed whitened through F
x, rng = rand_next(rng, Float64)        # pure: value and advanced generator
A = Vector{Float32}(undef, 1_000_000)
rng = rand_fill!(rng, A)                # threaded fill, same values as scalar draws
rand_at(rng, Float64, 17)               # 17th Float64 from here, without advancing

children = splitrng(rng, 1000)          # by index, from the key alone
rng, forks = forkrng(rng, 8)            # at the current step, parent moves on
proposals = subrng(rng, 1)              # by purpose id

die, rng = rand_next(rng, 1:6)          # bounded, Lemire, width from the range
z, rng = normal_next(rng)               # Box-Muller, bit identical to tandem-c
rng = normal_fill!(rng, A)              # threaded, pairs (z₀, z₁) from uniforms (2j−1, 2j)
rng = exponential_fill!(rng, A)         # −log(1 − u), one uniform per element

st = Stateful(42)                       # Random.AbstractRNG for rand, rand!, randn, ...
rand(st, 1:6)
```

## Derived draws

Bounded integers, normals, and exponentials follow Appendix A of [SPEC.md](SPEC.md), so
TandemRNG returns the values of tandem-c and tandem-cuda bit for bit. `Stateful`'s
`rand(st, 1:n)`, `rand!(st, A, 1:n)`, `randn`, `randn!`, `randexp`, and `randexp!` give the
same values as the immutable draws.

- **Bounded integers** use Lemire's method. A range of at most 2^32 values draws 32 bits,
  a larger one 64. The result type does not change the value, and `lo:hi` adds `lo` to a
  draw on `hi − lo + 1` values. A fill consumes one draw per element. A rejected draw
  retries on `splitrng(subrng(key at 0, P_w), g)` with `g` the draw's index in the stream,
  so a fill cut anywhere equals the whole fill. `rand_below_next` and `rand_below_fill!`
  name the width by the type of `n`, as tandem-c's `u32` and `u64` functions do.
- **Normals** are Box-Muller pairs `r cos 2πb`, `r sin 2πb` with `r = sqrt(−2 log(1 − a))`
  from uniform draws `2j − 1` and `2j`. A fill of `n` consumes `2·cld(n, 2)` uniforms.
  A scalar draw returns the cosine half and consumes two. Float32 normals are computed in
  Float32. `log`, `cos`, and `sin` are tandem-c's polynomials with explicit `fma`.
- **Exponentials** are `−log(1 − u)` with the same `log`, one uniform per element.
- Empty derived fills leave the position unchanged. Derived fills run on the CPU.

`test/derived.jl` checks the values against copies of tandem-c's `cross_below.h`,
`cross_normal.h`, and `cross_exponential.h` and tandem-cuda's `cross_fill_below.h`, and
the 1e6-element normal and exponential dumps of tandem-c against their SHA-256.

See [Derived draws](https://bjmcox.github.io/TandemRNG.jl/derived/) for the full rules.

GPU fills load with `using KernelAbstractions, GPUArraysCore` and any backend array:

```julia
using CUDA, KernelAbstractions, GPUArraysCore, MLDataDevices
rng = rng |> CUDADevice()
B = CUDA.zeros(Float32, 1 << 24)
rng = rand_fill!(rng, B)               # one work-item per chunk, equals the CPU fill
```

MLDataDevices is a core dependency. Binding preserves the key and position.
Allocating draws use the bound backend, and fills reject destinations on another backend,
including empty arrays. `rng |> CPUDevice()` restores CPU binding. Tokens name a backend;
select a physical GPU through the backend before allocation and execution.

`Stateful` caches the current row for sequential draws through the `Random` API.

Reactant supports compiled uniform draws, fills, random access, and static derivations.
Convert the generator with `Reactant.to_rarray(rng)` before compilation to keep its key
and position as runtime inputs. See [the integration guide](https://bjmcox.github.io/TandemRNG.jl/integrations/#reactant-integration).
TandemRNG provides an optional PureRNGs extension for uniform, normal, exponential,
and distribution draws, together with its split interface. PureRNGs owns the samplers.

## Precompilation

PrecompileTools caches scalar draws, fills, splitting, forking, and the `Random` bridge.
The core workload covers every supported chunk length and draw type.

Optional CUDA, AMDGPU, and Metal extensions precompile concrete GPU array methods when
installed with KernelAbstractions and GPUArraysCore. On Julia 1.12 or later, CUDA and
AMDGPU also compile small device fills when a working GPU is available. Metal coverage
is host-only. No GPU is required to load or precompile TandemRNG.

See the [precompilation notes](https://bjmcox.github.io/TandemRNG.jl/performance/#Precompilation) for coverage and version
limits. Use `benchmark/latency.jl` in fresh Julia processes to measure first-use latency.

## Speed

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
generators supply 24/53.

Apple M4, Julia 1.13.1 with 14 threads, 2026-10-04. Fills of 2^22 elements through the
`Random` API, best of five, GiB/s written. `Stateful` fills use every thread, Xoshiro one.
The one-task rows call the immutable fills with `nthreads = 1`.

| | `rand!` Float64 | `randn!` Float64 | `randn!` Float32 |
|---|---|---|---|
| Tandem `Stateful`, 14 tasks | 87.3 | 33.1 | 33.2 |
| Tandem, one task | 13.5 | 4.53 | 5.15 |
| Julia `Xoshiro`, one task | 16.1 | 7.22 | 1.26 |

The normals run tandem-c's polynomial Box-Muller, vectorized two doubles or four floats
wide with four interleaved iterations. Xoshiro's ziggurat is faster for one Float64 task.
tandem-c reaches 5.0 and 5.5 GiB/s for the one-task normal fills on the same machine.

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

Use the [public reproduction guide](benchmark/README.md) to benchmark supported result types.

## License and contact

Copyright 2026 Jessica Cox <jmcox@posteo.de>

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE).

Report bugs and request features through [GitHub issues](https://github.com/tandem-rng/TandemRNG.jl/issues).

**TandemRNG is not a cryptographic PRNG. Do not use it for cryptography or
security-sensitive applications.**
