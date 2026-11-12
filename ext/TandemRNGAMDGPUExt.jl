module TandemRNGAMDGPUExt

using TandemRNG
using AMDGPU
using KernelAbstractions
using GPUArraysCore
using PrecompileTools: @setup_workload, @compile_workload

const _ArrayFamilies = (AMDGPU.ROCArray{T,N,AMDGPU.Mem.HIPBuffer} where {T,N},)
const _Float64 = true
const _Backend = TandemRNG._AMDGPUBackend
const _DeviceExt = :AMDGPUExt
TandemRNG._backend_device(::_Backend) =
    TandemRNG.MLDataDevices.AMDGPUDevice(AMDGPU.device())
TandemRNG._allocate_array(::_Backend, ::Type{T}, dims::Dims) where {T} =
    AMDGPU.ROCArray{T}(undef, dims)
function _precompile_device()
    AMDGPU.functional() || return false
    try
        return !isempty(AMDGPU.devices())
    catch err
        # ROCm libraries can be installed on a machine without a GPU.
        err isa AMDGPU.HIP.HIPError &&
            err.code == AMDGPU.HIP.hipErrorNoDevice &&
            return false
        rethrow()
    end
end

_unsafe_free!(A) = AMDGPU.unsafe_free!(A)

include("gpu_precompile.jl")

end
