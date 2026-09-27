using Distributions, ForwardDiff

# Read slots bit by bit from a native byte fill. This oracle shares PureRNGs'
# mathematics but none of the bridge's slot extraction or cursor advancement.
struct ByteEngine{R}
    source::R
    bytes::Vector{UInt8}
end

oracle_slot(width) = first(filter(>=(width), (1, 8, 16, 32, 64)))
PR._engine_backend(engine::ByteEngine) = PR._engine_backend(engine.source)
PR._cursor_ordinal(::ByteEngine, bit::UInt64) = bit
PR._child_cursor(engine::ByteEngine, purpose::UInt64) = PR._child_cursor(engine.source, purpose)
PR._skip_takes(::ByteEngine, bit::UInt64, count::Integer, ::Val{W}) where {W} =
    bit + UInt64(count * oracle_slot(W))

function PR._take_bits(engine::ByteEngine, bit::UInt64, ::Val{W}) where {W}
    slot = oracle_slot(W)
    raw = UInt64(0)
    for offset = 0:(slot-1)
        position = Int(bit) + offset
        byte = engine.bytes[(position÷8)+1]
        raw |= UInt64((byte >> (position % 8)) & 1) << offset
    end
    return raw >> (slot - W), bit + UInt64(slot)
end

function check_codec(rng, codec, ::Type{T}, draw, fill!, at) where {T}
    count, width = PR._codec_takes(codec, T)
    slot = oracle_slot(PR._val_count(width))
    start = UInt64(cld(TR.rngposition(rng), slot) * slot)
    n = 129
    stop = start + UInt64(n * count * slot)
    source = TR.Tandem8x32{TR.chunk_length(rng)}(TR.rngkey(rng))
    bytes, _ = TR.rand_next(source, UInt8, Int(cld(stop, 8)))
    oracle = ByteEngine(source, bytes)
    expected = Vector{T}(undef, n)
    position = start
    for i in eachindex(expected)
        expected[i], position = PR._codec_take(codec, oracle, position, T)
    end
    @test position == stop
    scalar_rng = rng
    for value in expected
        actual, scalar_rng = draw(scalar_rng)
        @test isequal(actual, value)
    end
    @test TR.rngposition(scalar_rng) == stop
    for threaded in (false, true)
        destination = similar(expected)
        result, after = fill!(rng, destination, threaded)
        @test result === destination
        @test isequal(destination, expected)
        @test after == scalar_rng
    end
    @test isequal(at(rng, 1), first(expected))
    @test isequal(at(rng, n), last(expected))
    @test_throws ArgumentError at(rng, 0)
    @test_throws ArgumentError at(rng, big(2)^65)
end

# This destination has a valid length but no storage. A full-span check must
# reject before any read, write, or allocation proportional to that length.
struct HugeVector{T} <: AbstractVector{T} end
Base.size(::HugeVector) = (typemax(Int),)
Base.IndexStyle(::Type{<:HugeVector}) = IndexLinear()
Base.setindex!(::HugeVector, value, index) = error("write before span validation")
TR.MLDataDevices.get_device_type(::HugeVector) = TR.MLDataDevices.CPUDevice

struct HugeMatrix{T} <: AbstractMatrix{T} end
Base.size(::HugeMatrix) = (3, Int(0x0505050505050506))
Base.IndexStyle(::Type{<:HugeMatrix}) = IndexLinear()
Base.setindex!(::HugeMatrix, value, indices...) = error("write before span validation")
TR.MLDataDevices.get_device_type(::HugeMatrix) = TR.MLDataDevices.CPUDevice

@testset "PureRNGs engine conformance" begin
    @testset "normal and exponential slots" begin
        for K in (1, 32, 64), position in (1, 127, 1023, 1024K - 1)
            rng = TR.Tandem8x32{K}(TR.rngkey(TR.Tandem8x32(42)), position)
            for T in (Float16, Float32, Float64, Complex{Float16}, ComplexF32, ComplexF64)
                codec = PR._NormalCodec(PR._engine_backend(rng))
                check_codec(
                    rng,
                    codec,
                    T,
                    r -> PR.randn_next(r, T),
                    (r, a, t) -> PR.randn_next!(r, a; threaded = t),
                    (r, i) -> PR.randn_at(r, T, i),
                )
            end
            for T in (Float16, Float32, Float64)
                codec = PR._ExponentialCodec(PR._engine_backend(rng))
                check_codec(
                    rng,
                    codec,
                    T,
                    r -> PR.randexp_next(r, T),
                    (r, a, t) -> PR.randexp_next!(r, a; threaded = t),
                    (r, i) -> PR.randexp_at(r, T, i),
                )
            end
        end
    end

    @testset "distribution codecs" begin
        ext = Base.get_extension(PR, :PureRNGsDistributionsExt)
        for T in (Float32, Float64)
            distributions = (
                Normal(T(2), T(3)),
                Uniform(T(-1), T(2)),
                Exponential(T(2)),
                LogNormal(T(1), T(2)),
                Weibull(T(2), T(3)),
                Rayleigh(T(2)),
                Laplace(T(1), T(2)),
                Logistic(T(1), T(2)),
                Gumbel(T(1), T(2)),
                Pareto(T(2), T(3)),
                Frechet(T(2), T(3)),
                Cauchy(T(1), T(2)),
                TriangularDist(T(-1), T(2), T(0)),
                Bernoulli(T(0.3)),
                Gamma(T(0.2), T(3)),
                Gamma(T(5), T(2)),
                Chisq(T(3)),
                InverseGamma(T(2), T(3)),
                Beta(T(0.1), T(0.2)),
                TDist(T(3)),
                DiscreteUniform(-5, 17),
                DiscreteUniform(typemin(Int), typemax(Int)),
            )
            rng = TR.Tandem8x32(TR.rngkey(TR.Tandem8x32(99)), 32767)
            for d in distributions
                codec = ext._distribution_codec(d, PR._engine_backend(rng))
                check_codec(
                    rng,
                    codec,
                    ext._result_type(d),
                    r -> PR.rand_next(r, d),
                    (r, a, t) -> PR.rand_next!(r, d, a; threaded = t),
                    (r, i) -> PR.rand_at(r, d, i),
                )
            end
        end
    end

    @testset "allocation, views, and multivariate draws" begin
        rng = last(PR.rand_next(TR.Tandem8x32(42), Bool))
        normal, next = PR.randn_next(rng, Float32, 7, 5)
        @test PR.randn_next(rng, Float32, (7, 5)) == (normal, next)
        @test PR.randn_at(rng, Float32, 1:35) == vec(normal)
        backing = zeros(Float32, 14, 5)
        view = @view backing[1:2:end, :]
        @test last(PR.randn_next!(rng, view)) == next
        @test view == normal
        @test all(iszero, backing[2:2:end, :])
        @test PR.rand_next(rng, Uniform(0.0f0, 1.0f0), 35) == PR.rand_next(rng, Float32, 35)
        d = Dirichlet([0.2, 1.0, 3.0])
        a, after = PR.rand_next(rng, d, 7)
        @test all(isapprox.(sum(a; dims = 1), 1))
        @test a[:, 7] == PR.rand_at(rng, d, 7)
        @test TR.rngposition(after) == 64 + 7 * 3 * 17 * 64
        d = MvNormal([1.0, 2.0], [2.0, 3.0])
        a, after = PR.rand_next(rng, d, 7)
        @test a[:, 7] == PR.rand_at(rng, d, 7)
        @test TR.rngposition(after) == 64 + 7 * 2 * 64
        slope = ForwardDiff.derivative(x -> first(PR.rand_next(rng, Normal(x, 2.0))), 1.0)
        @test slope == 1
        dual = Normal(ForwardDiff.Dual(1.0, 1.0), 2.0)
        @test PR.rand_next(rng, dual, (2, 3)) == PR.rand_next(rng, dual, 2, 3)
        bits = falses(4097)
        window = @view bits[2:end]
        @test first(PR.rand_next!(rng, Bernoulli(0.3), window; threaded = true)) === window
        @test window == first(PR.rand_next(rng, Bernoulli(0.3), 4096))
        @test !bits[1]
        shape_slope =
            ForwardDiff.derivative(x -> first(PR.rand_next(rng, Gamma(x, 2.0))), 3.0)
        @test isfinite(shape_slope) && shape_slope > 0
    end

    @testset "checked fills and scalar wrapping" begin
        rng = TR.Tandem8x32(42)
        @test_throws ArgumentError PR.randn_next!(rng, HugeVector{Float32}())
        @test_throws ArgumentError PR.rand_next!(
            rng,
            Gamma(2.0, 3.0),
            HugeVector{Float64}(),
        )
        near_end = TR._advance(rng, typemax(UInt64) - UInt64(63))
        @test TR.rngposition(last(PR.randexp_next(near_end, Float64))) == 0
        expected = near_end
        for _ = 1:17
            _, expected = TR.rand_next(expected, UInt64)
        end
        @test last(PR.rand_next(near_end, Gamma(2.0, 3.0))) == expected
        destination = fill(-1.0, 1)
        @test_throws ArgumentError PR.randexp_next!(near_end, destination)
        @test_throws ArgumentError PR.rand_next!(near_end, Gamma(2.0, 3.0), destination)
        @test destination == [-1.0]
        misaligned_end = last(TR.rand_next(near_end, Bool))
        @test_throws ArgumentError PR.randn_next!(misaligned_end, Float64[])
        shifted = last(TR.rand_next(rng, Bool))
        @test TR.rngposition(last(PR.randn_next!(shifted, Float32[]))) == 32
        bits = falses(513)
        @test first(PR.rand_next!(rng, Bernoulli(0.3), bits; threaded = true)) === bits
        @test bits == first(PR.rand_next(rng, Bernoulli(0.3), 513))
    end

    @testset "Dirichlet column stream and span" begin
        for T in (Float32, Float64), K in (1, 32, 64), position in (1, 32767)
            source = TR.Tandem8x32{K}(42)
            rng = TR.Tandem8x32{K}(TR.rngkey(source), position)
            d = Dirichlet(T[0.2, 1, 3])
            codec = PR._DirichletCodec(d.alpha)
            count, width = PR._codec_takes(codec, T)
            slot = oracle_slot(PR._val_count(width))
            start = UInt64(cld(position, slot) * slot)
            stop = start + UInt64(9 * count * slot)
            bytes, _ = TR.rand_next(source, UInt8, Int(cld(stop, 8)))
            oracle = ByteEngine(source, bytes)
            expected = Matrix{T}(undef, 3, 9)
            cursor = start
            for column in axes(expected, 2)
                cursor = PR._column_take!(codec, oracle, cursor, expected, column)
            end
            @test cursor == stop
            for threaded in (false, true)
                destination = similar(expected)
                result, after = PR.rand_next!(rng, d, destination; threaded)
                @test result === destination
                @test destination == expected
                @test TR.rngposition(after) == stop
            end
            @test PR.rand_at(rng, d, 9) == expected[:, 9]
            @test PR.rand_at(rng, d, big(9)) == expected[:, 9]
            @test TR.rngposition(last(PR.rand_next(rng, d, 0))) == start
        end
        rng = TR.Tandem8x32(42)
        d = Dirichlet([0.2, 1.0, 3.0])
        @test_throws ArgumentError PR.rand_at(rng, d, UInt64(0x5555555555555557))
        @test_throws ArgumentError PR.rand_at(rng, d, big(2)^65)
        @test_throws ArgumentError PR.rand_next!(rng, d, HugeMatrix{Float64}())
        near_end = TR._advance(rng, typemax(UInt64) - UInt64(63))
        destination = fill(-1.0, 3, 2)
        @test_throws ArgumentError PR.rand_next!(near_end, d, destination)
        @test destination == fill(-1.0, 3, 2)
        jacobian = ForwardDiff.jacobian(d.alpha) do alpha
            first(PR.rand_next(rng, Dirichlet(alpha)))
        end
        @test all(isfinite, jacobian)
        @test all(abs.(sum(jacobian; dims = 1)) .< 1e-12)
    end
end
