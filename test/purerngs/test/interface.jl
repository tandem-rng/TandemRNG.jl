@testset "PureRNGs interface alignment" begin
    rng = last(TR.rand_next(TR.Tandem8x32(42), Bool))
    for threaded in (false, true), T in (Bool, Float32, ComplexF64, Char)
        @test PR.rand_next(rng, T, (3, 5); threaded) == TR.rand_next(rng, T, 3, 5)
        @test PR.rand_at(rng, T, 3:7; threaded) == [TR.rand_at(rng, T, i) for i = 3:7]
        @test isempty(PR.rand_at(rng, T, 0:-1; threaded))
    end
    @test PR.rand_next(rng, 3, 5; threaded = true) == TR.rand_next(rng, Float64, 3, 5)
    @test_throws ArgumentError PR.rand_next(rng, Float32, 3; threaded = 1)
    @test_throws ArgumentError PR.rand_at(rng, Float32, 0:3)
    @test_throws ArgumentError PR.rand_at(
        rng,
        UInt64,
        (UInt128(1)<<58):((UInt128(1)<<58)+1),
    )
    @test_throws ArgumentError PR.rand_at(rng, UInt64, (big(2)^128):(big(2)^128))
    population = (:left, :right, :middle)
    index, after = TR.rand_next(rng, 1:3)
    @test PR.rand_next(rng, population) == (population[index], after)
    @test first(PR.rand_next(rng, (2, 3))) isa Int
    @test_deprecated r"A bare tuple" PR.rand_next(rng, (2, 3))
    @test_throws ArgumentError PR.rand_next(rng, ())

    for (generate, next, set, fill!) in (
        (PR.randugen!!, PR.rand_next, PR.randuset!!, PR.rand_next!),
        (PR.randngen!!, PR.randn_next, PR.randnset!!, PR.randn_next!),
        (PR.randexpgen!!, PR.randexp_next, PR.randexpset!!, PR.randexp_next!),
    )
        @test generate(rng) == next(rng, Float64)
        @test generate(rng, Float32, (3, 5); threaded = true) ==
              next(rng, Float32, 3, 5; threaded = true)
        @test set(rng, 0.0f0) == next(rng, Float32)
        destination = zeros(Float32, 17)
        expected, after = next(rng, Float32, 17)
        actual, advanced = set(rng, destination)
        @test actual === destination
        @test (actual, advanced) == (expected, after)
        immutable_destination = 1.0:17.0
        replaced, advanced = set(rng, immutable_destination)
        @test replaced isa Vector
        @test length(replaced) == length(immutable_destination)
    end
    @test PR.randgen!!(rng, Float32, (3, 5)) == PR.rand_next(rng, Float32, 3, 5)
    @test PR.randset!!(rng, Float32, 0) == PR.rand_next(rng, Float32)
    destination = zeros(Float64, 17)
    actual, advanced = PR.randset!!(rng, Float32, destination)
    @test actual !== destination
    @test eltype(actual) === Float32
    @test (actual, advanced) == PR.rand_next(rng, Float32, 17)
    @test iszero(destination)

    distribution = Normal(2.0f0, 3.0f0)
    @test PR.randgen!!(rng, distribution, (3, 5)) == PR.rand_next(rng, distribution, 3, 5)
    destination = zeros(Float32, 17)
    actual, advanced = PR.randset!!(rng, distribution, destination)
    @test actual === destination
    @test (actual, advanced) == PR.rand_next(rng, distribution, 17)
    concentration = [0.2, 1.0, 3.0]
    saved = copy(concentration)
    actual, advanced = PR.randset!!(rng, Dirichlet(concentration), concentration)
    @test actual !== concentration
    @test concentration == saved
    @test (actual, advanced) == PR.rand_next(rng, Dirichlet(saved))
end

@testset "PureRNGs mutable sampler law" begin
    rng = last(TR.rand_next(TR.Tandem8x32{32}(42), Bool))
    bridge = PR.StatefulRNG(rng)
    @test bridge isa PR.StatefulRNG
    @test parent(bridge) == rng
    for T in (Bool, UInt128, Float16, Float32, Float64, ComplexF32, Char)
        expected, rng = PR.rand_next(rng, T)
        @test rand(bridge, T) == expected
        @test parent(bridge) == rng
    end
    for (draw, next) in ((randn, PR.randn_next), (randexp, PR.randexp_next))
        expected, rng = next(rng)
        @test draw(bridge) == expected
        @test parent(bridge) == rng
        for T in (Float16, Float32, Float64)
            expected, rng = next(rng, T)
            @test draw(bridge, T) == expected
            @test parent(bridge) == rng
        end
    end
    for (fill!, next!, destination) in (
        (rand!, PR.rand_next!, falses(17)),
        (rand!, PR.rand_next!, zeros(Float32, 17)),
        (randn!, PR.randn_next!, zeros(ComplexF32, 17)),
        (randexp!, PR.randexp_next!, zeros(Float16, 17)),
    )
        expected, rng = next!(rng, similar(destination))
        @test fill!(bridge, destination) === destination
        @test destination == expected
        @test parent(bridge) == rng
    end
    saved = copy(bridge)
    @test saved !== bridge
    @test saved == bridge
    @test Random.seed!(bridge, 99) === bridge
    @test parent(bridge) == TR.Tandem8x32{32}(99)
    @test copy!(bridge, saved) === bridge
    @test parent(bridge) == rng
    for seed in (nothing, RandomDevice())
        @test Random.seed!(bridge, seed) === bridge
        @test TR.chunk_length(parent(bridge)) == 32
        @test PR.rngposition(parent(bridge)) == 0
    end
    @test Random.seed!(bridge) === bridge
    for generate in (PR.randgen!!, PR.randugen!!, PR.randngen!!, PR.randexpgen!!)
        _, returned = generate(bridge, Float32)
        @test returned === bridge
    end
    # Loading PureRNGs never changes native Tandem's sampler policy.
    native = TR.Stateful(42)
    @test randn(native) == first(TR.normal_next(TR.Tandem8x32(42)))
end
