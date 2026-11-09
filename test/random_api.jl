# Check the Random API against the pure generator's values and successor state.

@testset "Random API" begin
    st = Stateful(42)
    pure = Tandem8x32(42)
    x, pure2 = rand_next(pure, Float64)
    @test rand(st) == x
    @test Tandem8x32(st) == pure2
    # A row change inside a draw loop, then a fill, then a draw: the wrapper stays on the
    # same stream as the pure chain throughout.
    for _ = 1:40
        rand(st)
        _, pure2 = rand_next(pure2, Float64)
    end
    @test Tandem8x32(st) == pure2
    @test rand(st, UInt32) == rand_next(pure2, UInt32)[1]

    Random.seed!(st, 42)
    A = rand!(st, zeros(Float64, 70))
    B = zeros(Float64, 70)
    rand_fill!(pure, B)
    @test A == B

    Random.seed!(st, 42)
    for T in (UInt32, Int64, Float32, Bool)
        v = rand(st, T)
        @test v isa T
    end
    @test all(in(1:10), rand(st, 1:10, 50))
    @test isfinite(randn(st))
    @test isfinite(randexp(st))

    st2 = Stateful{8}(42)
    @test chunk_length(Tandem8x32(st2)) == 8
    @test copy(st2) == st2 && copy(st2) !== st2
end

@testset "Random API: default derived draws" begin
    st = Stateful(42)
    expected, pure = normal_next(Tandem8x32(42))
    @test randn(st) == expected
    @test parent(st) == pure
    expected, pure = exponential_next(pure)
    @test randexp(st) == expected
    @test parent(st) == pure
end

@testset "Random API: copying and reseeding" begin
    source = Stateful{8}(42)
    rand(source, UInt64, 40)
    pure = parent(source)
    destination = Stateful{8}(99)
    @test copy!(destination, source) === destination
    @test parent(destination) == pure
    expected, pure = rand_next(pure, UInt32)
    @test rand(destination, UInt32) == expected
    Random.seed!(source, 7)
    expected, pure = rand_next(pure, Float64)
    @test rand(destination) == expected
    @test parent(destination) == pure

    for seed_args in ((), (nothing,), (Random.RandomDevice(),))
        @test Random.seed!(destination, seed_args...) === destination
        pure = parent(destination)
        @test chunk_length(pure) == 8
        @test rngposition(pure) == 0
        expected, pure = rand_next(pure, UInt64)
        @test rand(destination, UInt64) == expected
        @test parent(destination) == pure
    end
end
