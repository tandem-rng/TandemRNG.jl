@setup_workload let
    types = Base.uniontypes(DrawTypes)
    # Two groups at K=64 also exercise the joined task path with a small allocation.
    arrays = map(T -> Vector{T}(undef, 2 * 1024 * 64 ÷ draw_bits(T) + 3), types)
    @compile_workload begin
        for power = 0:16
            K = 1 << power
            rng = Tandem8x32{K}(42)
            Tandem8x32{K}(UInt128(42))
            Tandem8x32{K}(rngkey(rng), UInt64(128))
            st = Stateful(rng)
            Stateful{K}(42)
            for (T, A) in zip(types, arrays)
                _, shifted = rand_next(rng, T)
                rand_at(rng, T, 19)
                rand_fill!(rng, A; nthreads = 1)
                rand_fill!(shifted, A; nthreads = 2)
                rand(st, T)
                rand!(st, A)
            end
            splitrng(rng)
            splitrng(rng, Val(2))
            splitrng(rng, Val(8))
            splitrng(rng, 8)
            splitrng(rng, 1024)
            forkrng(rng)
            forkrng(rng, Val(2))
            forkrng(rng, Val(8))
            forkrng(rng, 8)
            forkrng(rng, 1024)
            subrng(rng, 7)
            subrng(rng, UInt64(7))
            randn(st)
            randexp(st)
            rand(st, 1:100)
            copy(st)
            Random.seed!(st, 77)
            precompile(Random.seed!, (Stateful{K},))
            Tandem8x32(st)
        end
        rng = Tandem8x32(42)
        st = Stateful(rng)
        for T in types
            matrix = Matrix{T}(undef, 17, 19)
            rand_next(rng, T, 17)
            rand_next(rng, T, (17, 19))
            rand_fill!(rng, matrix; nthreads = 1)
            rand_fill!(rng, view(matrix, :, 2); nthreads = 1)
            rand_fill!(rng, view(matrix, 1:2:17, 2); nthreads = 1)
            rand!(st, matrix)
            rand(st, T, 17)
            rand(st, T, 4, 4)
        end
        for T in (Float32, Float64)
            randn(st, T)
            randexp(st, T)
            randn(st, T, 17)
            randexp(st, T, 17)
            randn!(st, Vector{T}(undef, 17))
            randexp!(st, Vector{T}(undef, 17))
            normal_next(rng, T)
            exponential_next(rng, T)
            normal_fill!(rng, Vector{T}(undef, 17); nthreads = 1)
            exponential_fill!(rng, Vector{T}(undef, 17); nthreads = 1)
        end
        for U in (UInt32, UInt64)
            rand_below_next(rng, U(10))
            rand_below_fill!(rng, Vector{U}(undef, 17), U(10))
        end
        rand_next(rng, 1:6)
        rand_fill!(rng, Vector{Int}(undef, 17), 1:6)
        rand!(st, Vector{Int}(undef, 17), 1:6)
        Tandem8x32(42)
        Stateful(42)
        rand_next(rng)
        rand_next(rng, (3, 5))
        rand_next(rng, 3, 5)
        # Binding and host scalar arithmetic need no GPU driver.
        for K in (1, 32, 64),
            device in (
                MLDataDevices.CPUDevice(),
                MLDataDevices.CUDADevice(),
                MLDataDevices.AMDGPUDevice(),
                MLDataDevices.MetalDevice(),
            )

            bound = device(Tandem8x32{K}(42))
            for T in types
                rand_next(bound, T)
                rand_at(bound, T, 3)
            end
            splitrng(bound, Val(2))
            subrng(bound, 3)
            forkrng(bound, Val(2))
            Stateful(bound)
            MLDataDevices.CPUDevice()(bound)
        end
        # Compile entropy-seeded constructors without consuming operating-system entropy.
        precompile(Stateful, ())
    end
end
