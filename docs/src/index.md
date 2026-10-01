# TandemRNG.jl

TandemRNG provides immutable random number generators for CPU and GPU workloads.
A draw returns both its value and the next generator. Array fills and scalar draws
share the same stream across supported backends.

**TandemRNG is not a cryptographic PRNG. Do not use it for cryptography or
security-sensitive applications.**

```@example home
using TandemRNG
rng = Tandem8x32(42)
value, rng = rand_next(rng, Float64)
values, rng = rand_next(rng, Float32, 4)
(value, values)
```

Start with [Getting started](@ref). See [Streams and reproducibility](@ref) for
parallel jobs and saved state, [Devices](@ref) for GPU arrays, and
[Integrations](@ref) for Random, PureRNGs, and Reactant.
The [API reference](@ref) lists the exported interface.

## Current status

The default variant is `Tandem8x32-K32`. The package supports Julia 1.10 and
later, CPU arrays, and optional CUDA, AMDGPU, Metal, and Reactant integrations.
AMDGPU hardware validation remains open. Version 0.1.0 has no package release yet,
and the package is not registered.

The [validation evidence release](https://github.com/tandem-rng/TandemRNG.jl/releases/tag/statistical-evidence-2026-09-27)
contains statistical logs, case matrices, protocols, input hashes, frozen
reproduction scripts, and flagged results. Finite statistical tests do not prove
independence or cryptographic security.

The [algorithm specification](https://github.com/tandem-rng/TandemRNG.jl/blob/main/SPEC.md)
defines the recurrence, seeding, stream order, draw mappings, and test vectors.
See the [benchmark guide](https://github.com/tandem-rng/TandemRNG.jl/blob/main/benchmark/README.md)
to reproduce performance measurements.

## License and support

TandemRNG uses the [Apache License, Version 2.0](https://github.com/tandem-rng/TandemRNG.jl/blob/main/LICENSE).
Report bugs and request features through [GitHub issues](https://github.com/tandem-rng/TandemRNG.jl/issues).
