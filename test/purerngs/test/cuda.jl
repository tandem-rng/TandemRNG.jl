using CUDA

@testset "Tandem bridge CUDA residence" begin
    cpu = last(PR.rand_next(TR.Tandem8x32(42), Bool))
    device = TR.MLDataDevices.CUDADevice()
    to_cpu = TR.MLDataDevices.CPUDevice()
    rng = device(cpu)
    for active in CUDA.devices()
        CUDA.device!(active) do
            values = CUDA.zeros(Float32, 3)
            @test TR.MLDataDevices.get_device(rng) == TR.MLDataDevices.get_device(values)
            @test TR.MLDataDevices.get_device((rng, values)) ==
                  TR.MLDataDevices.get_device(values)
        end
    end
    values, after = PR.rand_next(rng, Float64, 17, 3)
    expected, expected_rng = PR.rand_next(cpu, Float64, 17, 3)
    @test values isa CuArray
    @test Array(values) == expected
    @test to_cpu(after) == expected_rng
    @test TR.MLDataDevices.get_device_type(after) == TR.MLDataDevices.CUDADevice
    @test to_cpu.(PR.splitrng(after, 3; threaded = false)) == PR.splitrng(expected_rng, 3)
    @test parent(PR.StatefulRNG(after)) == expected_rng

    for T in (Bool, UInt32), n in (0, 35)
        host = zeros(T, n)
        output = CuArray(host)
        @test_throws ArgumentError PR.rand_next!(cpu, output)
        @test_throws ArgumentError PR.rand_next!(rng, host)
        @test Array(output) == host
        @test_throws ArgumentError PR.rand_next!(rng, output; threaded = 1)
    end
    storage = CUDA.zeros(UInt32, 137)
    returned, after = PR.rand_next!(rng, view(storage, 1:2:137))
    expected = zeros(UInt32, 137)
    _, expected_rng = PR.rand_next!(cpu, view(expected, 1:2:137))
    @test parent(returned) === storage
    @test Array(storage) == expected
    @test to_cpu(after) == expected_rng
end

include("cuda_scalar.jl")
