# CPU benchmarks for scalar chains and bulk fills, with Xoshiro and Random123's Philox as
# references measured in the same process. Run from this directory's environment.
using TandemRNG
using BenchmarkTools
using Random
import Random123

const SUITE = BenchmarkGroup()

# Two complete K=32 groups include the reseeding cost for an immutable generator.
function draw_chain(rng::Tandem8x32, ::Type{T}, n) where {T}
    acc = T === Char ? UInt32(0) : zero(T)
    for _ = 1:n
        x, rng = TandemRNG.rand_next(rng, T)
        acc = T === Bool ? xor(acc, x) : T === Char ? xor(acc, UInt32(x)) : acc + x
    end
    return acc, rng
end

function draw_chain(rng::Random.AbstractRNG, ::Type{T}, n) where {T}
    acc = T === Char ? UInt32(0) : zero(T)
    # Random123's direct Float64 draw from UInt32 has only 32 random bits. The sampler
    # uses 52-bit extraction, as in its array fill.
    sampler = Random.Sampler(rng, T)
    for _ = 1:n
        x = rand(rng, sampler)
        acc = T === Bool ? xor(acc, x) : T === Char ? xor(acc, UInt32(x)) : acc + x
    end
    return acc
end

chain1024(rng) = draw_chain(rng, Float64, 1024)

# BenchmarkTools otherwise passes a runtime DataType and boxes the returned RNG.
typed_chain(f::F, rng, ::Val{T}, n) where {F,T} = f(rng, T, n)

SUITE["chain1024"] = BenchmarkGroup()
SUITE["chain1024"]["Tandem8x32{32}"] =
    @benchmarkable chain1024(rng) setup = (rng = Tandem8x32(1))
SUITE["chain1024"]["Tandem8x32{8}"] =
    @benchmarkable chain1024(rng) setup = (rng = Tandem8x32{8}(1))
SUITE["chain1024"]["Stateful"] = @benchmarkable chain1024(rng) setup = (rng = Stateful(1))
SUITE["chain1024"]["Xoshiro"] = @benchmarkable chain1024(rng) setup = (rng = Xoshiro(1))
for W in (UInt32, UInt64)
    SUITE["chain1024"]["Random123 Philox4x$(8sizeof(W))"] =
        @benchmarkable chain1024(rng) setup = (rng = Random123.Philox4x($W, (0, 1), 10))
end

SUITE["fill"] = BenchmarkGroup()
for T in (Float64, Float32, UInt32, Bool), n in (1 << 10, 1 << 20, 1 << 24)
    SUITE["fill"]["Tandem $T $n 1 thread"] =
        @benchmarkable TandemRNG.rand_fill!(rng, A; nthreads = 1) setup =
            (rng = Tandem8x32(1); A = Vector{$T}(undef, $n))
    SUITE["fill"]["Tandem $T $n"] = @benchmarkable TandemRNG.rand_fill!(rng, A) setup =
        (rng = Tandem8x32(1); A = Vector{$T}(undef, $n))
    SUITE["fill"]["Xoshiro $T $n"] =
        @benchmarkable rand!(rng, A) setup = (rng = Xoshiro(1); A = Vector{$T}(undef, $n))
    for W in (UInt32, UInt64)
        SUITE["fill"]["Random123 Philox4x$(8sizeof(W)) $T $n 1 thread"] =
            @benchmarkable rand!(rng, A) setup =
                (rng = Random123.Philox4x($W, (0, 1), 10); A = Vector{$T}(undef, $n))
    end
end

SUITE["derive"] = BenchmarkGroup()
SUITE["derive"]["splitrng 1024"] =
    @benchmarkable splitrng(rng, 1024) setup = (rng = Tandem8x32(1))
SUITE["derive"]["rand_at"] =
    @benchmarkable TandemRNG.rand_at(rng, Float64, 1000) setup = (rng = Tandem8x32(1))

fill_one!(rng::Tandem8x32, A) = TandemRNG.rand_fill!(rng, A; nthreads = 1)
fill_one!(rng::Random.AbstractRNG, A) = rand!(rng, A)
fill_parallel!(rng::Tandem8x32, A) = TandemRNG.rand_fill!(rng, A)

function cpu_cases(seed)
    return (
        ("Tandem", Tandem8x32(seed), fill_one!, draw_chain, fill_parallel!),
        (
            "Random123 Philox4x32",
            Random123.Philox4x(UInt32, (0, seed), 10),
            fill_one!,
            draw_chain,
            nothing,
        ),
        (
            "Random123 Philox4x64",
            Random123.Philox4x(UInt64, (0, seed), 10),
            fill_one!,
            draw_chain,
            nothing,
        ),
        ("Xoshiro", Xoshiro(seed), fill_one!, draw_chain, nothing),
    )
end

# `bytes` per timed call: the destination for a fill, sizeof(T) per draw for a chain.
function cpu_record(io, pass, operation, T, n, name, trial; bytes)
    best, middle = minimum(trial), median(trial)
    gibs(t) = bytes / 2.0^30 / (t.time / 1e9)
    println(
        io,
        join(
            (
                pass,
                operation,
                T,
                n,
                name,
                gibs(best),
                gibs(middle),
                best.memory,
                best.allocs,
                length(trial),
            ),
            '\t',
        ),
    )
    flush(io)
end

# Same-process comparisons use one preallocated array and alternate generator order.
# Mutable references advance normally. Construction, compilation, and first touch are untimed.
function compare_cpu(
    io::IO;
    sizes = (1 << 10, 1 << 20, 1 << 24),
    passes = 3,
    seconds = 0.3,
    samples = 100,
    seed = 42,
    threaded = true,
)

    passes > 0 && samples > 0 && isfinite(seconds) && seconds > 0 ||
        throw(ArgumentError("passes, samples, and seconds must be positive"))
    !isempty(sizes) && all(n -> n > 0, sizes) ||
        throw(ArgumentError("sizes must contain positive element counts"))
    generators = cpu_cases(seed)
    println(
        io,
        "# Julia $VERSION, $(Sys.CPU_NAME), max threads=$(Threads.nthreads()), Random123 $(pkgversion(Random123))",
    )
    println(
        io,
        "pass\toperation\ttype\telements\tgenerator\tbest_GiB_s\tmedian_GiB_s\tallocated_bytes\tallocations\tsamples",
    )
    for T in (Float64, Float32, UInt32, Bool), n in sizes
        A = Vector{T}(undef, n)
        expected = similar(A)
        after = TandemRNG.rand_fill!(Tandem8x32(seed), expected; nthreads = 1)
        for (_, rng, fill, _, _) in generators
            fill(rng, A)
        end
        for pass = 1:passes,
            (name, rng, fill, _, _) in (isodd(pass) ? generators : reverse(generators))

            trial = @benchmark $fill($rng, $A) evals = 1 samples = samples seconds = seconds
            cpu_record(io, pass, "fill_1task", T, n, name, trial; bytes = sizeof(A))
        end
        if threaded
            parallel = filter(case -> last(case) !== nothing, generators)
            fill_parallel!(Tandem8x32(seed), A) == after && A == expected ||
                error("threaded fill differs")
            for (_, rng, _, _, fill) in parallel
                fill(rng, A)
            end
            for pass = 1:passes,
                (name, rng, _, _, fill) in (isodd(pass) ? parallel : reverse(parallel))

                trial =
                    @benchmark $fill($rng, $A) evals = 1 samples = samples seconds = seconds
                cpu_record(io, pass, "fill_default", T, n, name, trial; bytes = sizeof(A))
            end
        end
    end
    chains = (
        (map(case -> (case[1], case[2], case[4]), generators))...,
        ("Tandem Stateful", Stateful(seed), draw_chain),
    )
    for T in (
        Bool,
        UInt8,
        Int8,
        UInt16,
        Int16,
        UInt32,
        Int32,
        UInt64,
        Int64,
        UInt128,
        Int128,
        Float16,
        Float32,
        Float64,
        Complex{Float16},
        Complex{Float32},
        Complex{Float64},
        Char,
    )
        n = max(1024, 2 * 32 * 1024 ÷ (T === Bool ? 1 : T === Char ? 64 : 8sizeof(T)))
        for (_, rng, chain) in chains
            chain(rng, T, n)
        end
        for pass = 1:passes, (name, rng, chain) in (isodd(pass) ? chains : reverse(chains))
            trial = @benchmark typed_chain($chain, $rng, $(Val(T)), $n) evals = 10 samples =
                samples seconds = seconds
            cpu_record(io, pass, "scalar_chain", T, n, name, trial; bytes = n * sizeof(T))
        end
    end
    return nothing
end
