# Integrations

## [Random](@id random-integration)

[`Stateful`](@ref) implements `Random.AbstractRNG`. It supports `rand`, `rand!`,
`randn`, `randexp`, ranges, and samplers built on the standard primitive draw methods.
Integer unit ranges and `randn`/`randexp` for Float32 and Float64 follow the
[derived draws](derived.md) of the specification. Julia's other collection and
range samplers retain their own algorithms and consumption rules.

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
chunk length. `StatefulRNG` aliases `Stateful`. `copy(rng)` duplicates the stream,
`copy!(destination, source)` copies its state, and `Random.seed!(rng, seed)` resets it.
No-argument, `nothing`, and `RandomDevice` reseeding obtain a fresh seed from the OS.
`parent(rng)` and `Tandem8x32(rng)` recover the current immutable CPU generator.
Constructing a wrapper from a device-bound generator explicitly binds it to the CPU.

Give each concurrent task its own mutable wrapper. Sharing one wrapper across tasks
does not provide synchronized stream advancement. Julia's higher-level samplers
need not retain identical output across Julia versions.

## PureRNGs

Loading both packages enables TandemRNG's PureRNGs extension, which requires
PureRNGs 0.0.2. PureRNGs 0.0.1 lacks the external wrapper hook.
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
| `StatefulRNG` | A CPU-bound PureRNGs wrapper using PureRNGs' sampler laws |
| `randgen!!`, `randset!!`, and their primitive variants | Pure continuations or mutable draws with PureRNGs' storage rules |

`PureRNGs.rand_next(rng, (2, 3))` samples the tuple as a population. Earlier
Tandem extensions treated that tuple as dimensions. Use
`PureRNGs.rand_next(rng, Float64, (2, 3))` for a matrix. Native
`TandemRNG.rand_next(rng, (2, 3))` retains its dimension shorthand.

The PureRNGs wrapper follows `PureRNGs.randn_next` and `randexp_next`, including
their result types and bit consumption. `TandemRNG.Stateful` keeps the native
Appendix A transforms. Loading PureRNGs does not change the native wrapper.
The PureRNGs wrapper also preserves pure uniform fills, including `BitArray`
bit order and position. Other collection and permutation methods use Julia's
generic `Random` samplers unless Tandem supplies a bridge method.

The bridge preserves alignment, final positions, allocation backend, and residence
checks. Bind with MLDataDevices before array calls. Allocating draws, destination fills and
`splitrng(rng, n)` default to `threaded = false`, matching PureRNGs. Pass
`threaded = true` to enable CPU threading. Native `rand_fill!` returns only the
generator. The native `rand_next!`, `randn_next!`, and `randexp_next!` aliases
return `(destination, next_rng)` and default to serial fills. Native `randn_next`
and `randexp_next` name the existing native normal and exponential draws.

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
general collection sampling, and range sampling retain PureRNGs' built-in generator APIs.
Scalar tuple picks use Tandem's bounded integer draw.

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
