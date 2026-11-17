get!(ENV, "XLA_REACTANT_GPU_PREALLOCATE", "false")

using Test, Random, Reactant
import PureRNGs as PR
import TandemRNG as TR

Reactant.set_default_backend(get(ENV, "TANDEM_REACTANT_BACKEND", "cpu"))

include("bridge.jl")

include("engine.jl")
include("interface.jl")

if get(ENV, "TANDEM_TEST_CUDA", "false") == "true"
    include("cuda.jl")
    include("device_engine.jl")
    TandemDeviceChecks.check(TR.MLDataDevices.CUDADevice(), CUDA.CuArray, CUDA.synchronize)
end

function compiled_bridge(rng, destination)
    bit, rng = PR.rand_next(rng, Bool)
    scalar, rng = PR.rand_next(rng, Float64)
    addressed = PR.rand_at(rng, UInt32, 3)
    _, rng = PR.rand_next!(rng, destination; threaded = false)
    allocated, rng = PR.rand_next(rng, Float32, 5, 3)
    return bit,
    scalar,
    addressed,
    rng,
    allocated,
    PR.splitrng(rng, Val(3)),
    PR.subrng(rng, 17),
    PR.subrng(rng, 3, UInt64(17), 1)
end

@testset "Tandem bridge through Reactant" begin
    rng = TR.Tandem8x32(42)
    input = Reactant.to_rarray(rng)
    output = Reactant.to_rarray(zeros(UInt16, 35))
    compiled = Reactant.@compile compiled_bridge(input, output)
    # Reuse the executable with a changed key and misaligned bit position.
    for base in (rng, last(TR.rand_next(TR.Tandem8x32(99), Bool)))
        destination = zeros(UInt16, 35)
        expected = compiled_bridge(base, destination)
        actual = compiled(Reactant.to_rarray(base), output)
        @test Bool(actual[1]) == expected[1]
        @test Float64(actual[2]) == expected[2]
        @test UInt32(actual[3]) == expected[3]
        @test TR.Tandem8x32(actual[4]) == expected[4]
        @test Array(actual[5]) == expected[5]
        @test map(TR.Tandem8x32, actual[6]) == expected[6]
        @test TR.Tandem8x32(actual[7]) == expected[7]
        @test TR.Tandem8x32(actual[8]) == expected[8]
        @test Array(output) == destination
    end
    @test_throws ArgumentError PR.rand_next(input, Normal())
end
