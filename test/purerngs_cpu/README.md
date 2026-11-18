# PureRNGs CPU bridge

This lean environment runs the shared bridge and mutable-wrapper contract tests
on Julia 1.10. It needs no CUDA or Reactant installation. The full conformance
project remains in `test/purerngs`.

From the TandemRNG repository root, develop Tandem locally and resolve PureRNGs
0.0.2 from the registry:

```sh
julia --startup-file=no --project=test/purerngs_cpu -e 'using Pkg; Pkg.develop(path="."); Pkg.test()'
```

Julia 1.10 ignores `[sources]`, so develop Tandem explicitly.
