# Changelog

## Unreleased

- The PureRNGs bridge supports normal, exponential, and distribution draws through
  PureRNGs' engine interface, including Dirichlet column fills. Sampler mathematics
  remain in PureRNGs. Native uniform streams retain their existing paths.
- Distribution fills and addressed draws check their complete input span before
  writing. Scalar draws retain Tandem's wrapping behavior.
- The PureRNGs bridge now defaults to `threaded = false` for `rand_next!` and
  `splitrng(rng, n)`, matching PureRNGs. Native Tandem methods retain their defaults.
- Short forward moves reuse cached state for distances of at most 1024 bits.
  Output streams and positions are unchanged.
