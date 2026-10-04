# Getting started

## Installation

Install from GitHub in your project environment:

```julia
using Pkg
Pkg.add(url = "https://github.com/tandem-rng/TandemRNG.jl")
```

Commit your project's `Project.toml` and `Manifest.toml` to retain the source revision.
TandemRNG requires Julia 1.10 or later. GPU packages are optional.

## Draw and continue

[`Tandem8x32`](@ref) is immutable. Assign the returned generator to continue its
stream. Reusing the same generator repeats the same draw.

```@example draws
using TandemRNG
start = Tandem8x32(42)
x, rng = rand_next(start, Float64)
y, rng = rand_next(rng, Float64)
(x, y, rngposition(rng))
```

[`rand_at`](@ref) reads without advancing. Its index is one-based and relative to
the current position.

```@example draws
@assert rand_at(start, Float64, 2) == y
rand_at(rng, Float64, 1)
```

Omitting the type selects `Float64`. Integer seeds must satisfy `0 <= seed < 2^128`.

## Allocate or fill arrays

Allocating draws return `(array, next_rng)`. Pass dimensions separately or as a tuple.
[`rand_fill!`](@ref) fills an existing array and returns only the next generator.

```@example arrays
using TandemRNG
start = Tandem8x32(42)
values, after_allocation = rand_next(start, Float32, 3, 2)
buffer = similar(values)
after_fill = rand_fill!(start, buffer)
@assert buffer == values
@assert rngposition(after_fill) == rngposition(after_allocation)
values
```

Arrays consume the stream in Julia's linear, column-major order. Fills agree with
successive scalar draws. CPU destinations require one-based indexing and can include
contiguous or strided views. CPU fills use Julia threads by default.
Pass `nthreads = 1` when an outer loop already distributes work.

## Result types

| Family | Supported types |
|:--|:--|
| Boolean | `Bool` |
| Unsigned integers | `UInt8`, `UInt16`, `UInt32`, `UInt64`, `UInt128` |
| Signed integers | `Int8`, `Int16`, `Int32`, `Int64`, `Int128` |
| Floating point | `Float16`, `Float32`, `Float64` |
| Complex | `Complex{Float16}`, `Complex{Float32}`, `Complex{Float64}` |
| Character | `Char` |

Floating-point draws lie in `[0, 1)`. Complex draws contain two successive real draws.
Character draws produce Unicode scalar values. Integers use their full bit width.
Device array restrictions appear in [Devices](@ref).

Integer ranges, normals, and exponentials have immutable draws and fills too, with the
values every Tandem port returns. See [Derived draws](derived.md).
Use [`Stateful`](@ref) with Julia's Random API for these and other samplers.
See [Random](@ref random-integration).
