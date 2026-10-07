# Check scalar draws in Metal kernels on Apple silicon through this environment: `Pkg.test()`
# is not wired to it, use `julia --project=test/metal test/metal/test/runtests.jl` or Kaimon's
# run_tests on this directory.

using TandemRNG
using Metal
using Test

# Work-item i draws one Float32 exponential at element i of the fill that starts at `start`.
function exponential_kernel!(out, key, start)
    i = thread_position_in_grid_1d()
    if i <= length(out)
        rng = Tandem8x32{32}(key, start + UInt64(32) * UInt64(i - 1))
        @inbounds out[i] = first(exponential_next(rng, Float32))
    end
    return nothing
end

@testset "metal: Float32 exponentials in a kernel equal the CPU fill" begin
    @test Metal.functional()
    n = 1 << 20
    for seed in (42, 2026), start in (0, 1, 77, 12345)
        key = rngkey(Tandem8x32(seed))
        cpu = Vector{Float32}(undef, n)
        exponential_fill!(Tandem8x32{32}(key, start), cpu)
        gpu = MtlVector{Float32}(undef, n)
        @metal threads = 256 groups = cld(n, 256) exponential_kernel!(gpu, key, UInt64(start))
        @test Array(gpu) == cpu
    end
end
