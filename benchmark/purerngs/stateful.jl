# Warm public Random APIs with fresh mutable state per sample. See ../README.md.
using BenchmarkTools, Random, SHA, Test
import Pkg, PureRNGs, TandemRNG

make_rng(::Val{:native}, seed) = TandemRNG.Stateful(TandemRNG.Tandem8x32(seed))
make_rng(::Val{:bridge}, seed) = PureRNGs.StatefulRNG(TandemRNG.Tandem8x32(seed))

function scalar_chain(rng, ::Val{T}, n) where {T}
    checksum = zero(T)
    for _ = 1:n
        checksum += rand(rng, T)
    end
    return checksum
end

function chain_trial(kind, tag::Val{T}, n, seed; seconds, samples) where {T}
    observed = Ref{Any}(nothing)
    checksum = Ref(zero(T))
    bench = @benchmarkable $checksum[] = scalar_chain(rng, $tag, $n) setup =
        (rng = make_rng($kind, $seed)) teardown =
        ($observed[] = ($checksum[], rand(rng, UInt32))) evals = 1
    trial = run(bench; seconds, samples)
    return trial, observed[]
end

function fill_trial(kind, A, seed; seconds, samples)
    observed = Ref{Any}(nothing)
    bench = @benchmarkable rand!(rng, $A) setup = (rng = make_rng($kind, $seed)) teardown =
        ($observed[] = (sum($A), rand(rng, UInt32))) evals = 1
    trial = run(bench; seconds, samples)
    return trial, observed[]
end

# These uniforms have the same bit extraction and stream ordering in both wrappers.
# Check the actual workload and its continuation before comparing its cost.
function validate(::Type{T}, n, seed) where {T}
    native, bridge = make_rng(Val(:native), seed), make_rng(Val(:bridge), seed)
    for _ = 1:n
        @assert isequal(rand(native, T), rand(bridge, T))
    end
    @assert parent(native) == parent(bridge)
    native, bridge = make_rng(Val(:native), seed), make_rng(Val(:bridge), seed)
    tag = Val(T)
    expected = @inferred scalar_chain(native, tag, n)
    actual = @inferred scalar_chain(bridge, tag, n)
    @assert isequal(actual, expected)
    @assert parent(native) == parent(bridge)
    native, bridge = make_rng(Val(:native), seed), make_rng(Val(:bridge), seed)
    A, B = Vector{T}(undef, n), Vector{T}(undef, n)
    @assert (@inferred rand!(native, A)) === A
    @assert (@inferred rand!(bridge, B)) === B
    @assert A == B
    @assert parent(native) == parent(bridge)
    return nothing
end

function source_hash(io, mod)
    root = pkgdir(mod)
    paths = [joinpath(root, "Project.toml")]
    for dir in ("src", "ext"), (path, _, names) in walkdir(joinpath(root, dir))
        for name in names
            endswith(name, ".jl") && push!(paths, joinpath(path, name))
        end
    end
    for path in sort!(paths)
        println(
            io,
            "# sha256 ",
            nameof(mod),
            "/",
            relpath(path, root),
            " ",
            bytes2hex(sha256(read(path))),
        )
    end
end

function record(io, pass, operation, T, n, kind, trial, observed)
    best, middle = minimum(trial), median(trial)
    println(
        io,
        join(
            (
                pass,
                operation,
                T,
                n,
                kind,
                best.time,
                middle.time,
                best.memory,
                best.allocs,
                length(trial),
                observed...,
            ),
            '\t',
        ),
    )
    flush(io)
    return middle.time / n
end

function compare_stateful(
    io::IO = stdout;
    sizes = (1024, 1 << 20),
    passes = 3,
    seconds = 0.3,
    samples = 100,
    seed = 42,
)
    !isempty(sizes) &&
        all(>(0), sizes) &&
        passes > 0 &&
        samples > 0 &&
        isfinite(seconds) &&
        seconds > 0 || throw(ArgumentError("benchmark settings must be positive"))
    println(
        io,
        "# Julia $VERSION, $(Sys.CPU_NAME), threads=$(Threads.nthreads()), seed=$seed",
    )
    for mod in (TandemRNG, PureRNGs, BenchmarkTools)
        println(io, "# ", nameof(mod), " ", pkgversion(mod))
    end
    info = Pkg.dependencies()[Base.PkgId(PureRNGs).uuid]
    println(io, "# PureRNGs revision=$(info.git_revision), tree=$(info.tree_hash)")
    println(io, "# runner sha256=", bytes2hex(sha256(read(@__FILE__))))
    source_hash(io, TandemRNG)
    source_hash(io, PureRNGs)
    println(
        io,
        "# settings sizes=$sizes, passes=$passes, seconds=$seconds, samples=$samples, evals=1",
    )
    println(
        io,
        "pass\toperation\ttype\telements\twrapper\tbest_ns\tmedian_ns\tallocated_bytes\tallocations\tsamples\tchecksum\tnext_UInt32",
    )
    ranges = Dict{Tuple{Symbol,DataType,Int,Symbol},Vector{Float64}}()
    for T in (UInt32, Float32, Float64), n in sizes
        validate(T, n, seed)
        A = zeros(T, n)
        kinds = (Val(:native), Val(:bridge))
        # Compilation, first touch, wrapper construction, and validation are untimed.
        for kind in kinds
            scalar_chain(make_rng(kind, seed), Val(T), n)
            rand!(make_rng(kind, seed), A)
        end
        for pass = 1:passes, kind in (isodd(pass) ? kinds : reverse(kinds))
            name = kind == Val(:native) ? :native : :bridge
            for operation in (:chain, :fill)
                trial, observed =
                    operation == :chain ?
                    chain_trial(kind, Val(T), n, seed; seconds, samples) :
                    fill_trial(kind, A, seed; seconds, samples)
                time = record(io, pass, operation, T, n, name, trial, observed)
                push!(get!(ranges, (operation, T, n, name), Float64[]), time)
            end
        end
    end
    println(io, "# Repeated median ranges in ns/element:")
    for (key, times) in sort!(collect(ranges); by = x -> string(first(x)))
        println(io, "# ", join(key, '\t'), '\t', minimum(times), '\t', maximum(times))
    end
    return ranges
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 2 || throw(ArgumentError("usage: stateful.jl [passes] [seconds]"))
    passes = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
    seconds = length(ARGS) < 2 ? 0.3 : parse(Float64, ARGS[2])
    compare_stateful(; passes, seconds)
end
