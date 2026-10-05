# One-task fills of uniforms, normals, and exponentials against Xoshiro and the Philox
# generators, and the multithreaded Tandem fills, for the tables of docs/src/performance.md:
#
#     julia --startup-file=no --threads=14 --project=benchmark benchmark/draws.jl [log2 n] [passes]
#
# Each figure is the minimum over BenchmarkTools samples of one fill, then the best of the
# passes. Passes alternate the generator order. Rates count the bytes written.
using TandemRNG, BenchmarkTools, Random
using PureRNGs: PureRNGs, Philox4x32
import Random123

const DRAWS = (:rand, :randn, :randexp)

tandem1(::Val{:rand}) = (rng, A) -> rand_fill!(rng, A; nthreads = 1)
tandem1(::Val{:randn}) = (rng, A) -> normal_fill!(rng, A; nthreads = 1)
tandem1(::Val{:randexp}) = (rng, A) -> exponential_fill!(rng, A; nthreads = 1)
pure1(::Val{:rand}) = (rng, A) -> PureRNGs.rand_next!(rng, A; threaded = false)
pure1(::Val{:randn}) = (rng, A) -> PureRNGs.randn_next!(rng, A; threaded = false)
pure1(::Val{:randexp}) = (rng, A) -> PureRNGs.randexp_next!(rng, A; threaded = false)
random(::Val{:rand}) = rand!
random(::Val{:randn}) = randn!
random(::Val{:randexp}) = randexp!

# (name, generator, fill) for one task, every generator offering the draw.
function one_task(d, seed)
    return (
        ("Tandem", Tandem8x32(seed), tandem1(Val(d))),
        ("Xoshiro", Xoshiro(seed), random(Val(d))),
        ("PureRNGs Philox4x32", Philox4x32(seed), pure1(Val(d))),
        ("Random123 Philox4x32", Random123.Philox4x(UInt32, (0, seed), 10), random(Val(d))),
    )
end

many_tasks(d, seed) = (("Tandem Stateful", Stateful(seed), random(Val(d))),)

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
    println("draw\ttype\tgenerator\tGiB/s")
    rows = Tuple{String,DataType,String,Float64}[]
    for d in DRAWS, T in (d === :rand ? (Float64, Float32, UInt32) : (Float64, Float32))
        A = Vector{T}(undef, 2^log2n)
        groups = threaded ? (one_task(d, 42), many_tasks(d, 42)) : (one_task(d, 42),)
        for cases in groups, (name, rate) in rates(cases, A, passes)
            push!(rows, ("$(d)!", T, name, round(rate; digits = 2)))
            println(join(rows[end], '\t'))
        end
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse.(Int, ARGS)...)
end
