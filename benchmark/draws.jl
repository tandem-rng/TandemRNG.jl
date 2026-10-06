# One-task fills of uniforms, normals, and exponentials against Xoshiro and Random123's
# Philox4x32, and the multithreaded `Stateful` fills against one Xoshiro or Philox4x32 per
# task, for the tables of docs/src/performance.md:
#
#     julia --startup-file=no --threads=14 --project=benchmark benchmark/draws.jl [log2 n] [passes]
#
# Each figure is the minimum over BenchmarkTools samples of one fill, then the best of the
# passes. Passes alternate the generator order. Rates count the bytes written.
using TandemRNG, BenchmarkTools, Random
import Random123

const DRAWS = (:rand, :randn, :randexp)

tandem1(::Val{:rand}) = (rng, A) -> rand_fill!(rng, A; nthreads = 1)
tandem1(::Val{:randn}) = (rng, A) -> normal_fill!(rng, A; nthreads = 1)
tandem1(::Val{:randexp}) = (rng, A) -> exponential_fill!(rng, A; nthreads = 1)
random(::Val{:rand}) = rand!
random(::Val{:randn}) = randn!
random(::Val{:randexp}) = randexp!

# (name, generator, fill) for one task, every generator offering the draw.
function one_task(d, seed)
    return (
        ("Tandem", Tandem8x32(seed), tandem1(Val(d))),
        ("Xoshiro", Xoshiro(seed), random(Val(d))),
        ("Random123 Philox4x32", Random123.Philox4x(UInt32, (0, seed), 10), random(Val(d))),
    )
end

# The references have no threaded fill, so each task fills its own contiguous chunk with its
# own generator.
struct PerTask{R}
    rngs::Vector{R}
end

function per_task(fill)
    return function (p::PerTask, A::Array)
        n, len = length(p.rngs), length(A)
        GC.@preserve A Threads.@sync for t = 1:n
            lo, hi = div((t - 1) * len, n), div(t * len, n)
            # An Array chunk keeps the references on their dense-array fill paths.
            Threads.@spawn fill(p.rngs[t], unsafe_wrap(Array, pointer(A, lo + 1), hi - lo))
        end
        return A
    end
end

function many_tasks(d, seed)
    n = Threads.nthreads()
    return (
        ("Tandem Stateful", Stateful(seed), random(Val(d))),
        ("Xoshiro", PerTask([Xoshiro(seed + t) for t = 1:n]), per_task(random(Val(d)))),
        (
            "Random123 Philox4x32",
            PerTask([Random123.Philox4x(UInt32, (t, seed), 10) for t = 1:n]),
            per_task(random(Val(d))),
        ),
    )
end

function rates(cases, A, passes)
    best = Dict{String,Float64}()
    for (_, rng, fill) in cases
        fill(rng, A)
    end
    for pass = 1:passes, (name, rng, fill) in (isodd(pass) ? cases : reverse(cases))
        t = minimum(@benchmark $fill($rng, $A) evals = 1 samples = 100 seconds = 0.3).time
        best[name] = max(get(best, name, 0.0), sizeof(A) / 2.0^30 / (t / 1e9))
    end
    return best
end

function main(log2n = 22, passes = 3; threaded = true)
    println("# Julia $VERSION, $(Sys.CPU_NAME), $(Threads.nthreads()) threads, 2^$log2n elements")
    println("draw\ttype\ttasks\tgenerator\tGiB/s")
    rows = Tuple{String,DataType,Int,String,Float64}[]
    for d in DRAWS, T in (d === :rand ? (Float64, Float32, UInt32) : (Float64, Float32))
        A = Vector{T}(undef, 2^log2n)
        groups = threaded ? ((1, one_task(d, 42)), (Threads.nthreads(), many_tasks(d, 42))) :
            ((1, one_task(d, 42)),)
        for (tasks, cases) in groups, (name, rate) in rates(cases, A, passes)
            push!(rows, ("$(d)!", T, tasks, name, round(rate; digits = 2)))
            println(join(rows[end], '\t'))
        end
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse.(Int, ARGS)...)
end
