"""
    TandemRNG

Tandem: an asymmetric duplex random number generator designed for GPUs. Eight 32-bit words,
a hidden bijective clock that keys a Philox-shaped multiplicative Feistel on the exposed half,
chunked reseeding for splitting and random access, and a bit-aligned stream law with
no straddling scalar component draws. The algorithm specification is [tandem-spec](https://github.com/tandem-rng/spec/blob/main/SPEC.md).
"""
module TandemRNG

using Random
import MLDataDevices
using PrecompileTools: @setup_workload, @compile_workload

export Tandem8x32, Stateful
export rand_next, rand_at, rand_fill!
export rand_below_next, rand_below_fill!
export normal_next, normal_fill!, exponential_next, exponential_fill!
export splitrng, forkrng, subrng
export rngkey, rngposition, chunk_length

# The PureRNGs bridge and Reactant extension share this runtime state carrier.
struct _ReactantRNG{K,A}
    state::A
end

include("vector.jl")
include("core.jl")
include("devices.jl")
include("generator.jl")
include("split.jl")
include("fill.jl")
include("derived.jl")
include("random.jl")
include("precompile.jl")

end
