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
