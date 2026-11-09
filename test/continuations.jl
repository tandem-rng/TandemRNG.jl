@testset "native continuation aliases" begin
    rng = rand_next(Tandem8x32(42), Bool)[2]
    stateful = StatefulRNG(rng)
    @test (rand(stateful), parent(stateful)) == rand_next(rng)

    expected = Matrix{UInt32}(undef, 3, 3)
    expected_rng = rand_fill!(rng, expected; nthreads = 1)
    destination = similar(expected)
    returned, next_rng = rand_next!(rng, destination)
    @test returned === destination
    @test (returned, next_rng) == (expected, expected_rng)
    @test rand_next!(rng, destination; threaded = true) == (expected, expected_rng)

    for (draw, fill, native_draw, native_fill) in (
        (randn_next, randn_next!, normal_next, normal_fill!),
        (randexp_next, randexp_next!, exponential_next, exponential_fill!),
    )
        @test draw(rng) == native_draw(rng)
        for T in (Float32, Float64)
            @test draw(rng, T) == native_draw(rng, T)
            expected = Matrix{T}(undef, 3, 3)
            expected_rng = native_fill(rng, expected; nthreads = 1)
            destination = similar(expected)
            returned, next_rng = fill(rng, destination)
            @test returned === destination
            @test (returned, next_rng) == (expected, expected_rng)
            @test fill(rng, destination; threaded = true) == (expected, expected_rng)
            @test draw(rng, T, (3, 3)) == (expected, expected_rng)
            @test draw(rng, T, 3, 3; threaded = true) == (expected, expected_rng)
        end
        @test draw(rng, (3, 3)) == draw(rng, Float64, (3, 3))
        @test draw(rng, 3, 3; threaded = true) == draw(rng, Float64, (3, 3))
    end
end
