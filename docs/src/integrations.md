# Integrations

## [Random](@id random-integration)

[`Stateful`](@ref) implements `Random.AbstractRNG`. It supports `rand`, `rand!`,
`randn`, `randexp`, ranges, and samplers built on the standard primitive draw methods.
Integer ranges, collections, `randn`, and `randexp` for Float32 and Float64 follow the
[derived draws](derived.md) of the specification, so they equal the immutable draws
and the other Tandem ports.

```@example random
using TandemRNG, Random
rng = Stateful(42)
die = rand(rng, 1:6)
normal = randn(rng)
values = rand(rng, Float32, 4)
immutable_rng = parent(rng)
(die, normal, values, rngposition(immutable_rng))
```

`Stateful()` obtains a seed from `RandomDevice`. `Stateful{K}(seed)` selects the
chunk length. `copy(rng)` duplicates the stream, and `Random.seed!(rng, seed)` resets it.
`parent(rng)` and `Tandem8x32(rng)` recover the current immutable CPU generator.
Constructing a wrapper from a device-bound generator explicitly binds it to the CPU.

Give each concurrent task its own mutable wrapper. Sharing one wrapper across tasks
does not provide synchronized stream advancement. Julia's higher-level samplers
need not retain identical output across Julia versions.

## PureRNGs

Loading both packages enables TandemRNG's PureRNGs extension.
Use qualified names to avoid conflicts between their exported functions.

```julia
import PureRNGs, TandemRNG
rng = TandemRNG.Tandem8x32(42)
values, rng = PureRNGs.rand_next(rng, Float64, 17, 3)
buffer = similar(values)
buffer, rng = PureRNGs.rand_next!(rng, buffer; threaded = false)
stateful = PureRNGs.StatefulRNG(rng)
```

| PureRNGs operation | Tandem behavior |
|:--|:--|
| `rand_next`, `rand_at` | Uniform scalar, addressed, and allocating draws |
| `rand_next!` | Destination fill returning `(destination, next_rng)` |
| `randn_next`, `randn_next!`, `randn_at` | Normal draws |
| `randexp_next`, `randexp_next!`, `randexp_at` | Exponential draws |
| `rand_next(rng, distribution, ...)` | Distribution draws using PureRNGs' samplers |
| `rngkey`, `rngposition` | Current immutable state |
| `splitrng`, `subrng` | Child derivations with the same stream rules |
| `StatefulRNG` | A CPU-bound `TandemRNG.Stateful` |

The bridge preserves alignment, final positions, allocation backend, and residence
checks. Bind with MLDataDevices before array calls. Destination fills and
`splitrng(rng, n)` default to `threaded = false`, matching PureRNGs. Pass
`threaded = true` to enable CPU threading. Native `rand_fill!` returns only the
generator.

Load Distributions to use its supported samplers through PureRNGs:

```julia
using Distributions
normal, rng = PureRNGs.randn_next(rng, Float32)
values, rng = PureRNGs.rand_next(rng, Gamma(2f0, 3f0), 1024)
columns, rng = PureRNGs.rand_next(rng, Dirichlet([0.2, 1.0, 3.0]), 16)
```

PureRNGs owns the sampler mathematics and differentiation rules. Tandem supplies
bits through its native row generator. Normal, exponential, mapped univariate,
Gamma-family, MvNormal, and Dirichlet draws use this interface. Categorical,
collection sampling, and range sampling retain PureRNGs' built-in generator APIs.

Each sampler take uses one aligned Tandem slot: one bit for a one-bit take,
otherwise the smallest of 8, 16, 32, and 64 bits that holds it. The sampler reads
the slot's most significant bits. `rngposition` remains a bit position. Native
uniform draws retain their existing stream law.

Scalar draws may wrap at the end of Tandem's stream. Array fills, including
single-column Dirichlet draws, and addressed draws reject a span that passes the
end. Fills check the whole span before writing, including multiplication overflow.
An empty fill still aligns the position to its slot.

Device fills use the generator's bound device. Metal rejects Float64 sampler
results before kernel compilation. Transcendental results need not be identical
across backends. PureRNGs' Enzyme rules for device fills do not cover Tandem fills.
Large-chunk fills retain recurrence state across shared-memory tiles instead of
regenerating each tile's prefix. Small fills keep the single-tile path.
CUDA stages small Dirichlet columns in shared memory for coalesced access.
The normalization mathematics and component order remain in PureRNGs.
Tiny Dirichlet concentrations use PureRNGs' scaled normalization on CUDA and Metal.
Recovery rereads the held bits without changing parent advancement.

Use `TandemRNG.forkrng` for forks. The uniform and static derivation methods also
work with converted Reactant states. The new distribution methods do not support
Reactant states.

## [Reactant](@id reactant-integration)

Reactant 0.2.280 or later in the 0.2 series enables the optional extension.
Select a backend, convert the generator and destination, then compile:

```julia
using TandemRNG, Reactant
Reactant.set_default_backend("cpu")
rng = Reactant.to_rarray(Tandem8x32(42))
destination = Reactant.to_rarray(zeros(Float32, 1024))
compiled_fill = Reactant.@compile rand_fill!(rng, destination)
rng = compiled_fill(rng, destination)
values = Array(destination)
cpu_rng = Tandem8x32(rng)
```

The converted state carries its key and position as runtime data. The executable
accepts changed keys and positions with matching chunk length, backend, and array
shape. `Tandem8x32(rng)` restores the CPU generator, including positions above `2^63`.

Compiled operations include `rand_next`, one-based `rand_at`, `rand_fill!`,
`splitrng`, `forkrng`, and `subrng`. Child counts use `Val(N)`. Addressed indices
and purpose IDs must be static. Supported result types appear in [Devices](@ref).
Fills replace the traced array's value and preserve CPU stream order and final position.

Bulk fills seed each covered chunk once. Scalar draws reconstruct their block,
so prefer fills for many draws. Compiled operations omit exhaustion checks.
Keep the final bit position below `2^64`.
