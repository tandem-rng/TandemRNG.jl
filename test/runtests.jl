using TandemRNG
using Test
using Aqua
using Random

@testset "TandemRNG.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(TandemRNG)
    end
    include("stream_law.jl")
    include("result_types.jl")
    include("derive.jl")
    include("random_api.jl")
    include("derived.jl")
    include("choice.jl")
    include("devices.jl")
    include("statistics.jl")
end
