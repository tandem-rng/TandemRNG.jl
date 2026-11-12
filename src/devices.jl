# Backend tokens follow PureRNGs' residence contract. A token does not pin a
# physical device. Select the active device through the backend before allocating.
struct _CPUBackend end
struct _CUDABackend end
struct _AMDGPUBackend end
struct _MetalBackend end

const _BackendToken = Union{_CPUBackend,_CUDABackend,_AMDGPUBackend,_MetalBackend}
const _GPUBackend = Union{_CUDABackend,_AMDGPUBackend,_MetalBackend}

MLDataDevices.get_device_type(::_CPUBackend) = MLDataDevices.CPUDevice
MLDataDevices.get_device_type(::_CUDABackend) = MLDataDevices.CUDADevice
MLDataDevices.get_device_type(::_AMDGPUBackend) = MLDataDevices.AMDGPUDevice
MLDataDevices.get_device_type(::_MetalBackend) = MLDataDevices.MetalDevice

_backend_device(token::_BackendToken) = MLDataDevices.get_device_type(token)()

@inline function _check_fill_device(rng, destination)
    generator_device = MLDataDevices.get_device_type(rng.device)
    destination_device = MLDataDevices.get_device_type(destination)
    generator_device <: destination_device || throw(
        ArgumentError(
            "destination device differs from the generator device: generator on " *
            "$generator_device, destination on $destination_device",
        ),
    )
    return nothing
end

@inline _check_serviceability(rng, ::Type) = nothing
@inline _allocate_array(::_CPUBackend, ::Type{T}, dims::Dims) where {T} =
    Array{T}(undef, dims)

@noinline function _allocate_array(device::_BackendToken, ::Type, ::Dims)
    throw(
        ArgumentError(
            "load the GPU backend to allocate on $(MLDataDevices.get_device_type(device))",
        ),
    )
end

function _allocate_draw_array(rng, ::Type{T}, dims::Dims) where {T}
    all(>=(0), dims) || throw(ArgumentError("dimensions must be non-negative"))
    _check_serviceability(rng, T)
    return _allocate_array(rng.device, T, dims)
end
