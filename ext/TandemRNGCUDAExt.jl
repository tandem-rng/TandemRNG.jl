module TandemRNGCUDAExt

using TandemRNG
using CUDA
using KernelAbstractions
using GPUArraysCore
using PrecompileTools: @setup_workload, @compile_workload

const _ArrayFamilies = (
    CUDA.CuArray{T,N,CUDA.DeviceMemory} where {T,N},
    CUDA.CuArray{T,N,CUDA.UnifiedMemory} where {T,N},
)
const _Float64 = true
const _Backend = TandemRNG._CUDABackend
const _DeviceExt = :CUDAExt
TandemRNG._backend_device(::_Backend) = TandemRNG.MLDataDevices.CUDADevice(CUDA.device())
TandemRNG._allocate_array(::_Backend, ::Type{T}, dims::Dims) where {T} =
    CUDA.CuArray{T}(undef, dims)
_precompile_device() = CUDA.functional() && !isempty(CUDA.devices())

_unsafe_free!(A) = CUDA.unsafe_free!(A)

include("gpu_precompile.jl")

end
