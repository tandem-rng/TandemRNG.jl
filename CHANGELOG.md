# Changelog

## Unreleased

- GPU PureRNGs fills reuse recurrence state across shared-memory tiles for long
  chunks. Small fills and the default chunk length retain their existing path.
- CUDA Dirichlet batches stage small columns in shared memory for coalesced
  access, retaining the existing normalization order and draw values.
- GPU Dirichlet fills recover finite normalized draws when every log-Gamma value
  overflows. Recovery reuses the held bits and works on CUDA and Metal.
- The PureRNGs bridge supports normal, exponential, and distribution draws through
  PureRNGs' engine interface, including Dirichlet column fills. Sampler mathematics
  remain in PureRNGs. Native uniform streams retain their existing paths.
- Distribution fills and addressed draws check their complete input span before
  writing. Scalar draws retain Tandem's wrapping behavior.
- The PureRNGs bridge now defaults to `threaded = false` for `rand_next!` and
  `splitrng(rng, n)`, matching PureRNGs. Native Tandem methods retain their defaults.
- Short forward moves reuse cached state for distances of at most 1024 bits.
  Output streams and positions are unchanged.
