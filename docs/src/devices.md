# Devices

## Bind a generator

TandemRNG uses MLDataDevices in its core interface. New generators are CPU-bound.
Bind a generator before allocating or filling arrays on another backend.
Install the backend together with KernelAbstractions and GPUArraysCore. For CUDA:

```julia
using Pkg
Pkg.add(["CUDA", "KernelAbstractions", "GPUArraysCore", "MLDataDevices"])
```

```julia
using TandemRNG, MLDataDevices
using CUDA, KernelAbstractions, GPUArraysCore
rng = Tandem8x32(42) |> CUDADevice()
values, rng = rand_next(rng, Float32, 1024)
rng = rand_fill!(rng, values)
CUDA.synchronize()
host_values = Array(values)
cpu_rng = rng |> CPUDevice()
```

Binding preserves the key, position, and cached stream state. Draws and child
derivations retain the binding. Allocating draws use the bound backend. Destination
fills reject a backend mismatch before mutation, including for empty arrays.
Unsupported operations do not fall back to CPU execution.

Tokens identify a backend, not a physical GPU. Select the physical device through
the backend, such as `CUDA.device!`, before allocation and execution.
Keep the active device and arrays consistent.

`MLDataDevices.get_device(rng)` returns the bound backend device. With CUDA or
AMDGPU loaded, it resolves the active physical device at query time, as PureRNGs
does. `get_device((rng, values))` therefore composes with arrays on that device.

## Backend support

| Backend | Binding | Array draw types |
|:--|:--|:--|
| CPU | `CPUDevice()` | All 18 types |
| CUDA | `CUDADevice()` | All except `UInt128` and `Int128` |
| AMDGPU | `AMDGPUDevice()` | All except `UInt128` and `Int128` |
| Metal | `MetalDevice()` | All except `UInt128`, `Int128`, `Float64`, and `ComplexF64` |
| Reactant | `Reactant.to_rarray(rng)` | Bool, 8–64-bit integers, and Float16/32/64 |

Load AMDGPU or Metal in place of CUDA for those backends. Load GPUArrays as well
for Metal's optional precompile extension. Supported versions are CUDA 5.8 or later
in 5.x, CUDA 6.x, AMDGPU 2.x, and Metal 1.7 or later in 1.x.
AMDGPU hardware validation remains open.

GPU fills are asynchronous. Synchronize before timing completion or reading results
from the host. CPU fill thread settings do not select GPU threads.

## Scalar calls and mutable state

Host scalar calls run on the host even when the generator is GPU-bound.
Scalar calls inside a GPU kernel run on the device. A backend token does not move
a host call into a kernel.

[`Stateful(rng)`](@ref Stateful) explicitly rebinds its state to the CPU.
Use immutable generators for device work. See [Reactant](@ref reactant-integration)
for compilation with runtime keys and positions.
