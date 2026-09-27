get!(ENV, "XLA_REACTANT_GPU_PREALLOCATE", "false")

using Test, Random, Reactant
import PureRNGs as PR
import TandemRNG as TR

Reactant.set_default_backend(get(ENV, "TANDEM_REACTANT_BACKEND", "cpu"))

@testset "PureRNGs result type parity" begin
    types = (
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
    for T in types
        @test first(PR.rand_next(PR.Philox4x32(42), T)) isa T
        rng = last(TR.rand_next(TR.Tandem8x32(42), Bool))
        @test (@inferred PR.rand_next(rng, T)) == TR.rand_next(rng, T)
        @test PR.rand_next(rng, T, 17, 3) == TR.rand_next(rng, T, 17, 3)
        @test PR.rand_at(rng, T, 19) == TR.rand_at(rng, T, 19)
    end
end

@testset "Tandem bridge" begin
    rng = TR.Tandem8x32(42)
    bit, shifted = PR.rand_next(rng, Bool)
    @test (bit, shifted) == TR.rand_next(rng, Bool)
    values, next_rng = PR.rand_next(shifted, Float64, 17, 3)
    expected = similar(values)
    expected_rng = TR.rand_fill!(shifted, expected; nthreads = 1)
    @test values == expected
    @test (PR.rngkey(next_rng), PR.rngposition(next_rng)) ==
          (TR.rngkey(expected_rng), TR.rngposition(expected_rng))
    destination = zeros(UInt32, 35)
    returned, after = PR.rand_next!(next_rng, destination; threaded = false)
    expected = similar(destination)
    expected_rng = TR.rand_fill!(next_rng, expected; nthreads = 1)
    @test returned === destination
    @test destination == expected
    @test after == expected_rng
    @test PR.rand_at(shifted, Float64, 17) == values[17]

    @test PR.splitrng(after) == TR.splitrng(after)
    @test PR.splitrng(after, Val(3)) == TR.splitrng(after, Val(3))
    @test PR.splitrng(after, 3; threaded = false) == TR.splitrng(after, 3)
    @test PR.splitrng(after, 3) == TR.splitrng(after, 3)
    @test PR.subrng(after, 17) == TR.subrng(after, 17)

    bridge = PR.StatefulRNG(after)
    value, expected_rng = TR.rand_next(after, Float32)
    @test rand(bridge, Float32) == value
    @test parent(bridge) == expected_rng
    @test PR.rngposition(rng) == 0
end

include("engine.jl")

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
    PR.subrng(rng, 17)
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
        @test Array(output) == destination
    end
    @test_throws ArgumentError PR.rand_next(input, Normal())
end
