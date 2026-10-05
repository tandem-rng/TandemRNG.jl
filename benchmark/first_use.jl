# This file runs once per engine in a fresh process with warm package caches.
using Printf

const engine = only(ARGS)
engine in ("tandem", "random123-32", "random123-64", "xoshiro") ||
    throw(ArgumentError("unknown engine: $engine"))

function report(label, stats)
    compile = hasproperty(stats, :compile_time) ? stats.compile_time : NaN
    @printf("%s\t%.9f\t%d\t%.9f\n", label, stats.time, stats.bytes, compile)
    flush(stdout)
    return stats.value
end

println("operation\tseconds\tallocated_bytes\tcompile_seconds")
if engine == "tandem"
    report("load", @timed @eval import Random, TandemRNG)
elseif startswith(engine, "random123")
    report("load", @timed @eval import Random, Random123)
else
    report("load", @timed @eval import Random)
end

rng = report(
    "construct",
    @timed if engine == "tandem"
        TandemRNG.Tandem8x32(42)
    elseif startswith(engine, "random123")
        Random123.Philox4x(endswith(engine, "32") ? UInt32 : UInt64, (0, 42), 10)
    else
        Random.Xoshiro(42)
    end
)

for T in (
    Float64,
    Float32,
    Float16,
    UInt128,
    Int128,
    UInt64,
    Int64,
    UInt32,
    Int32,
    UInt16,
    Int16,
    UInt8,
    Int8,
    Bool,
    Complex{Float16},
    Complex{Float32},
    Complex{Float64},
    Char,
)
    A = Vector{T}(undef, 32771)
    if engine == "tandem"
        report("$T/next", @timed TandemRNG.rand_next(rng, T))
        report("$T/fill", @timed TandemRNG.rand_fill!(rng, A; nthreads = 1))
    else
        # Match Random123's array precision instead of its direct 32-bit Float64 draw.
        sampler = Random.Sampler(rng, T)
        report("$T/next", @timed Random.rand(rng, sampler))
        report("$T/fill", @timed Random.rand!(rng, A))
    end
end
