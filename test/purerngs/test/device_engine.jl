module TandemDeviceChecks

using Test, Distributions, KernelAbstractions
import TandemRNG as TR, PureRNGs as PR
const GX = Base.get_extension(TR, :TandemRNGPureRNGsKernelAbstractionsExt)

@kernel function child_words!(destination, engine)
    child, cursor = PR._child_cursor(engine, UInt64(7))
    for i in eachindex(destination)
        value, cursor = PR._take_bits(child, cursor, Val(64))
        @inbounds destination[i] = value
    end
end

function check_dirichlet_recovery(device, array, synchronize; types = (Float32, Float64))
    @testset "Tiny Dirichlet recovery across a chunk boundary" begin
        source = TR.Tandem8x32(TR.rngkey(TR.Tandem8x32(42)), 130999)
        rng = device(source)
        for T in types
            tiny = T === Float32 ? T(1e-40) : T(1e-320)
            distribution = Dirichlet(tiny .* T[1, 2, 3])
            expected, next = PR.rand_next(source, distribution, 3)
            destination = array(zeros(T, 3, 3))
            _, after = PR.rand_next!(rng, distribution, destination)
            synchronize()
            @test Array(destination) == expected
            @test TR.rngposition(after) == TR.rngposition(next)
            @test Array(PR.rand_at(rng, distribution, 3)) == expected[:, 3]
        end
    end
end

function check(device, array, synchronize; types = (Float32, Float64))
    @testset "Device engine" begin
        check_dirichlet_recovery(device, array, synchronize; types)
        seed = TR.Tandem8x32(42)
        for T in types, position in (1, 32767)
            rng = device(TR.Tandem8x32(TR.rngkey(seed), position))
            for d in (
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
                Gamma(T(0.2), T(3)),
                Gamma(T(5), T(2)),
                Chisq(T(3)),
                InverseGamma(T(2), T(3)),
                Beta(T(0.1), T(0.2)),
                TDist(T(3)),
            )
                destination = array(zeros(T, 129))
                result, after = PR.rand_next!(rng, d, destination)
                synchronize()
                expected = Vector{T}(undef, length(destination))
                state = rng
                for i in eachindex(expected)
                    expected[i], state = PR.rand_next(state, d)
                end
                @test result === destination
                @test after == state
                @test isapprox(
                    Array(destination),
                    expected;
                    rtol = 64eps(T),
                    atol = 64eps(T),
                )
            end
        end
        rng = device(seed)
        for T in types, (components, columns) in ((3, 257), (16, 257), (513, 3))
            d = Dirichlet(fill(T(0.2), components))
            destination = array(zeros(T, components, columns))
            result, after = PR.rand_next!(rng, d, destination)
            synchronize()
            expected = Matrix{T}(undef, components, columns)
            codec = PR._DirichletCodec(d.alpha)
            state = rng
            for column in axes(expected, 2)
                state = PR._column_take!(codec, rng, state, expected, column)
            end
            @test result === destination
            @test after == state
            @test isapprox(Array(destination), expected; rtol = 64eps(T), atol = 64eps(T))
        end
        @testset "Dirichlet strided destination" begin
            d = Dirichlet(fill(0.2f0, 17))
            storage = array(fill(NaN32, 34, 257))
            destination = @view storage[1:2:34, :]
            _, after = PR.rand_next!(rng, d, destination)
            expected, next = PR.rand_next(rng, d, 257)
            synchronize()
            @test Array(destination) == Array(expected)
            @test after == next
            @test all(isnan, Array(storage)[2:2:34, :])
        end
        for T in (Float16, Float32, Complex{Float16}, ComplexF32), n in (0, 1, 63, 64, 65)
            destination = array(zeros(T, n))
            _, after = PR.randn_next!(rng, destination)
            synchronize()
            expected = Vector{T}(undef, n)
            state = rng
            for i in eachindex(expected)
                expected[i], state = PR.randn_next(state, T)
            end
            @test after == state
            F = typeof(real(zero(T)))
            @test isapprox(Array(destination), expected; rtol = 8eps(F), atol = 8eps(F))
        end
        for K in (1, 32, 64)
            bound = device(TR.Tandem8x32{K}(TR.rngkey(seed)))
            backend = PR._engine_backend(bound)
            engine = GX._TileEngine{K,typeof(backend)}(bound.key, backend)
            words = array(zeros(UInt64, 1025))
            child_words!(get_backend(words), 1)(words, engine; ndrange = 1)
            synchronize()
            child = TR.subrng(TR.Tandem8x32{K}(TR.rngkey(seed)), 7)
            @test Array(words) == first(TR.rand_next(child, UInt64, length(words)))
        end
        # Exercise complete and partial input tiles, including a native chunk
        # group larger than the shared tile. Compare public fills and positions.
        for K in (1, 64, 256), position in (0, 130999)
            source = TR.Tandem8x32{K}(TR.rngkey(seed), position)
            bound = device(source)
            for (d, n) in (
                (Normal(2.0f0, 3.0f0), 4096),
                (Uniform(-1.0f0, 2.0f0), 4096),
                (Exponential(2.0f0), 4096),
                (TriangularDist(-1.0f0, 2.0f0, 0.0f0), 4096),
                (Gamma(0.2f0, 3.0f0), 259),
                (Beta(0.1f0, 0.2f0), 259),
                (Bernoulli(0.3f0), 4097),
            )
                expected, next = PR.rand_next(source, d, n)
                destination = array(similar(expected))
                _, after = PR.rand_next!(bound, d, destination)
                synchronize()
                @test TR.rngposition(after) == TR.rngposition(next)
                @test isapprox(
                    Array(destination),
                    expected;
                    rtol = 64eps(Float32),
                    atol = 64eps(Float32),
                )
            end
        end
        # Force the cold Gamma child path instead of waiting for eight rejections.
        codec = PR._GammaCodec(0.2f0, 3.0f0, PR._engine_backend(rng), 0)
        destination = array(zeros(Float32, 129))
        _, after = PR._engine_fill!(rng, destination, false, codec)
        synchronize()
        expected = Vector{Float32}(undef, 129)
        state = rng
        for i in eachindex(expected)
            expected[i], state = PR._engine_draw_next(state, codec, Float32)
        end
        @test after == state
        @test isapprox(
            Array(destination),
            expected;
            rtol = 64eps(Float32),
            atol = 64eps(Float32),
        )
        @test_throws ArgumentError PR.randn_next!(rng, Float32[])
        if device isa TR.MLDataDevices.MetalDevice
            @test_throws ArgumentError PR.randn_next(rng, Float64, 0)
        end
    end
end

end
