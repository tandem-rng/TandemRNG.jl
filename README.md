<p align="center"><img src="docs/src/assets/lockup.png" width="560" alt="tandem rng .jl"></p>

# TandemRNG

[![Build Status](https://github.com/tandem-rng/TandemRNG.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tandem-rng/TandemRNG.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/tandem-rng/TandemRNG.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tandem-rng/TandemRNG.jl)
[![Documentation](https://img.shields.io/badge/docs-guide-blue.svg)](https://bjmcox.github.io/TandemRNG.jl/)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10%2B-9558B2?logo=julia)](https://julialang.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

Reference Julia implementation of [Tandem8x32](https://github.com/tandem-rng/spec/blob/main/SPEC.md),
a random number generator built to be fast on CPUs and GPUs alike. Fills and scalar
draws give the same stream on the CPU, CUDA, AMDGPU, Metal and Reactant.

**TandemRNG is not a cryptographic PRNG. Do not use it for cryptography or
security-sensitive applications.**

Install from GitHub using Julia 1.10 or later. The package is not registered yet.

```julia
using Pkg
Pkg.add(url="https://github.com/tandem-rng/TandemRNG.jl")
```

```julia
using TandemRNG

rng = Tandem8x32(42)                    # chunks of 32 steps, seed whitened through F
x, rng = rand_next(rng, Float64)        # pure: value and advanced generator
A = Vector{Float32}(undef, 1_000_000)
rng = rand_fill!(rng, A)                # threaded fill, same values as scalar draws
children = splitrng(rng, 1000)          # by index, from the key alone
z, rng = normal_next(rng)               # ziggurat, bit identical to tandem-c
i, rng = choice_next(rng, ChoiceTable([1, 2, 3, 4]))   # weighted index, alias table
```

The [documentation](https://bjmcox.github.io/TandemRNG.jl/) covers derived draws, devices,
integrations, speed, and the [validation evidence](https://github.com/tandem-rng/TandemRNG.jl/releases/tag/statistical-evidence-2026-09-27).
[`benchmark/README.md`](benchmark/README.md) reproduces the measurements.

Portions of the code were generated with the assistance of LLMs.

[Documentation](https://bjmcox.github.io/TandemRNG.jl/) · [Apache 2.0 license](LICENSE) · [Issues](https://github.com/tandem-rng/TandemRNG.jl/issues)
