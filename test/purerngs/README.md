# PureRNGs bridge conformance

Use Julia 1.13 with sibling `TandemRNG.jl` and `PureRNGs.jl` checkouts.
TandemRNG owns the bridge. PureRNGs needs no Tandem-specific extension.
The relative paths in `Project.toml` select those sources.

From the TandemRNG root, run:

```sh
julia --project=test/purerngs -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The default suite checks the CPU bridge, sampler engine contract, checked spans,
ForwardDiff rules, and Reactant on its CPU backend. The engine tests compare
sampler results with an independent reader of native Tandem bytes. Set
`TANDEM_TEST_CUDA=true` to add CUDA residence, sampler, and device-kernel checks. Set
`TANDEM_REACTANT_BACKEND=gpu` to run the Reactant checks on a supported GPU.
CUDA and Reactant must use compatible devices in that combined run.
Device checks include tiny Dirichlet concentrations across a native chunk boundary.
The normalization hook rereads the held bits through PureRNGs without advancing the parent stream.

`test/device_engine.jl` also exposes `TandemDeviceChecks.check` for Metal validation
in an environment containing Metal. Use `Metal.MtlArray`, `Metal.synchronize`,
`MLDataDevices.MetalDevice()`, and `types = (Float32,)`.
