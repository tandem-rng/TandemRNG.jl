using TandemRNG.MLDataDevices

struct UnsupportedDevice <: MLDataDevices.AbstractDevice end

@testset "device residence" begin
    cpu = last(rand_next(Tandem8x32(42), Bool))
    for device in (CPUDevice(), CUDADevice(), AMDGPUDevice(), MetalDevice())
        bound = @inferred device(cpu)
        @test MLDataDevices.get_device(bound)(bound) == bound
        @test CPUDevice()(bound) == cpu
        @test chunk_length(typeof(bound)) == chunk_length(cpu)
        for T in (Bool, UInt32, Float64)
            expected, expected_rng = rand_next(cpu, T)
            actual, next_rng = @inferred rand_next(bound, T)
            @test actual == expected
            @test CPUDevice()(next_rng) == expected_rng
            @test device isa MLDataDevices.get_device_type(next_rng)
        end
        children = @inferred splitrng(bound, 3; threaded = false)
        @test CPUDevice().(children) == splitrng(cpu, 3)
        @test all(child -> device isa MLDataDevices.get_device_type(child), children)
        parent_rng, forks = forkrng(bound, Val(3))
        expected_parent, expected_forks = forkrng(cpu, Val(3))
        @test CPUDevice()(parent_rng) == expected_parent
        @test map(CPUDevice(), forks) == expected_forks
        @test all(child -> device isa MLDataDevices.get_device_type(child), forks)
        child = subrng(bound, 17)
        @test CPUDevice()(child) == subrng(cpu, 17)
        @test device isa MLDataDevices.get_device_type(child)
        @test parent(Stateful(bound)) == cpu
    end
    values, after = @inferred rand_next(cpu, Float64, 17, 3)
    @test MLDataDevices.get_device((cpu, values)) == MLDataDevices.get_device(values)
    expected = similar(values)
    @test rand_fill!(cpu, expected) == after
    @test values == expected
    @test_throws ArgumentError UnsupportedDevice()(cpu)
    @test_throws ArgumentError rand_next(cpu, Float32, -1)
    @test_throws ArgumentError rand_next(MetalDevice()(cpu), Float64, 0)

    for device in (CUDADevice(), AMDGPUDevice(), MetalDevice()), n in (0, 17)
        destination = fill(UInt32(0xfeed), n)
        @test_throws ArgumentError rand_fill!(device(cpu), destination)
        @test destination == fill(UInt32(0xfeed), n)
    end
end
