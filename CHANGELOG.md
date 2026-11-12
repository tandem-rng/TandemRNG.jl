# Changelog

## Unreleased

- `MLDataDevices.get_device(rng)` matches PureRNGs' device lookup and composes with
  arrays. CUDA and AMDGPU resolve the active physical device at query time.
- Native `Stateful` default normal and exponential draws now use the same Float64
  algorithms as their typed forms. It supports `copy!` and all standard reseed forms.
- `StatefulRNG` aliases `Stateful`. Native `rand_next!`, `randn_next`, `randn_next!`,
  `randexp_next`, and `randexp_next!` provide continuation names with native stream laws.
- Uniform addressed draws reject alignment and span overflow, including complex pairs.
  Scalar continuation draws retain their wrapping behavior.
- The PureRNGs extension requires 0.0.2 and uses its mutable sampler bridge. Normal,
  exponential, and packed-bit draws now match PureRNGs continuations through that bridge.
  Native `Stateful` keeps Tandem's sampler laws. Unified `!!` adapters support immutable
  Tandem generators and preserve PureRNGs' destination replacement and alias rules.
- PureRNGs uniform allocations accept `threaded`, and uniform addressed draws accept
  contiguous index ranges. A bare tuple now means a population in `PureRNGs.rand_next`.
  Replace `PureRNGs.rand_next(rng, dims)` with `PureRNGs.rand_next(rng, Float64, dims)`
  when `dims` is a tuple shape. Native Tandem's tuple shape shorthand stays available.
- Float32 exponentials round to within 0.58 ulp, bit identical to tandem-c 1c75956 on the
  CPU and in Metal kernels, so every uniform on the 2^-24 grid maps back to itself. This
  changes Float32 exponential values. Uniforms, Float32 normals, Float64 exponentials and
  positions are unchanged.
- Scalar draws that start aligned and end inside their block take one branch. The 8- to
  32-bit scalar chains run about 40 % faster and the 64-bit chains about 10 % faster.
  Float64 fills on AArch64 convert with one fixed-point instruction. Values are unchanged.
- AVX2 fills regain the speed they lost to the NEON row-loop rewrite: about twice as fast
  for UInt32 and Float32 on Zen 2. Values are unchanged.
- Scalar draws compile inside Metal kernels. The out-of-line state builders take the key as
  two 64-bit words, since Metal has no 128-bit integers.
- Float64 normals are the 1024-layer ziggurat of Appendix A, one UInt64 draw per element,
  bit identical to tandem-c. This changes every Float64 normal value and the positions
  after Float64 normal draws. An empty Float64 normal fill aligns the position to 64 bits.
  Float32 normals are unchanged.
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
