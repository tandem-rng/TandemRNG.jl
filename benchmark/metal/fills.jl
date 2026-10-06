# Tandem fills on the Apple GPU against Metal.jl's own `rand!`, which draws these types from
# Metal Performance Shaders' Philox, for the Metal rows of docs/src/performance.md:
#
#     julia --startup-file=no --project=benchmark/metal benchmark/metal/fills.jl [passes]
#
# Each case warms for 0.5 s, then takes the minimum of seven synchronized fills. Passes
# alternate the generator order. Rates count the bytes written.
using TandemRNG, Metal, Random

const CASES = ((UInt32, 24), (UInt32, 26), (UInt64, 25), (Float32, 26))

function fill_time(fill!, A)
    stop = time_ns() + 500_000_000
    while time_ns() < stop
        Metal.@sync fill!(A)
    end
    return minimum(1:7) do _
        t = time_ns()
        Metal.@sync fill!(A)
        time_ns() - t
    end
end

function main(passes = 3)
    rng = Tandem8x32(42) |> TandemRNG.MLDataDevices.MetalDevice()
    generators = (
        ("Tandem", A -> rand_fill!(rng, A)),
        ("Metal.jl rand! (MPS Philox)", A -> rand!(A)),
    )
    println("# Julia $VERSION, Metal $(pkgversion(Metal)), $(Metal.device().name)")
    println("type\telements\tgenerator\tGiB/s")
    for (T, log2n) in CASES
        A = MtlArray{T}(undef, 2^log2n)
        best = Dict{String,Float64}()
        for pass = 1:passes, (name, fill!) in (isodd(pass) ? generators : reverse(generators))
            rate = sizeof(A) / 2.0^30 / (fill_time(fill!, A) / 1e9)
            best[name] = max(get(best, name, 0.0), rate)
        end
        for (name, _) in generators
            println(join((T, "2^$log2n", name, round(best[name]; digits = 1)), '\t'))
        end
        Metal.unsafe_free!(A)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse.(Int, ARGS)...)
end
