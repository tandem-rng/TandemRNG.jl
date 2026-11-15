# PureRNGs CPU bridge

This lean environment runs the shared bridge and mutable-wrapper contract tests
on Julia 1.10. It needs no CUDA or Reactant installation. The full conformance
project remains in `test/purerngs`.

From the TandemRNG repository root, clone current PureRNGs dev beside the checkout
and develop both sources in one resolution:

```sh
git clone --depth 1 --branch dev https://github.com/BJMCox/PureRNGs.jl ../PureRNGs.jl
julia --startup-file=no --project=test/purerngs_cpu -e 'using Pkg; Pkg.develop([PackageSpec(path="."), PackageSpec(path="../PureRNGs.jl")]); Pkg.test()'
```

Julia 1.10 ignores `[sources]`, so the explicit source paths are required.
Use an existing PureRNGs checkout by replacing its path in the command.
The clone command tracks `dev`; record its commit when retaining test results.
