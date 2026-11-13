# PureRNGs bridge conformance

Use Julia 1.13 or later with the `TandemRNG.jl` checkout.
TandemRNG owns the bridge. PureRNGs needs no Tandem-specific extension.
`Project.toml` tracks PureRNGs' `dev` branch by URL and uses the local Tandem source.
The registered 0.0.1 release lacks the external wrapper hook. This project requires
0.0.2 from `dev`. The CPU CI job refreshes PureRNGs before testing, so it tests the
latest `dev` head rather than the registered release.

From the TandemRNG root, run:

```sh
julia --project=test/purerngs -e 'using Pkg; Pkg.instantiate(); Pkg.update("PureRNGs"); Pkg.test()'
```

To test a local PureRNGs checkout instead, run `Pkg.develop(path = "/path/to/PureRNGs.jl")`
in this test project before testing. Restore the URL source in `Project.toml` and
remove the local manifest to return to `dev` tracking.

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
