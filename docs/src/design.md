# Design and validation

Tandem is a random number generator built to be fast on CPUs and GPUs alike: **T**wo-half
**A**symmetric **N**onlinear **D**uplex with **E**volving **M**ultipliers.
The [specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) defines the
algorithms, stream order, draw mappings, and test vectors.

- Eight 32-bit words, using 32x32→64 multiplication, xor, rotate, and add.
- A hidden half runs a bijective clock and receives feedback from the new exposed half.
  The exposed half is a Philox-shaped Feistel whose multipliers and masks come from the hidden half.
- The stream is cut into chunks of `K` steps. A keyed seeding function starts every chunk, so
  splitting and forking cost one seeding per pair of children. Random access adds at most
  `K` steps after seeding.
- Chunks come in groups of eight, and the stream is a sequence of 128-byte rows: the eight
  16-byte blocks of a group at one step. On a GPU each work-item owns one chunk and eight
  adjacent work-items write one contiguous row per step, direct 16-byte stores, no shared
  memory. On a CPU the eight chunks step together, with one eight-lane vector per state word.
- A bit-aligned stream law. Scalar components stay inside 128-bit blocks. Complex draws
  compose two real draws. Fills and scalar loops agree on every supported backend.
- All 18 PureRNGs result types: Bool, 8–128-bit integers, Float16/32/64, their complex
  types, and Char. Device fills exclude 128-bit integers. Metal also excludes Float64
  and ComplexF64. Reactant supports Bool, 8–64-bit integers, and Float16/32/64.
- The transport form of a generator is its 128-bit key plus a 64-bit bit position.

## Validation evidence

The [validation evidence](https://github.com/tandem-rng/TandemRNG.jl/releases/tag/statistical-evidence-2026-09-27)
contains logs, case matrices, protocols, input hashes, frozen reproduction scripts,
and flagged results from PractRand, BigCrush, HWD, gjrand, and reduced-round tests.

- [Main validation records](https://github.com/tandem-rng/TandemRNG.jl/releases/download/statistical-evidence-2026-09-27/tandemrng-validation-records-20260926.7z)
- [Reduced-round margin records](https://github.com/tandem-rng/TandemRNG.jl/releases/download/statistical-evidence-2026-09-27/tandemrng-validation-margin-20260927.7z)
- [Validation audit](https://github.com/tandem-rng/TandemRNG.jl/releases/download/statistical-evidence-2026-09-27/tandemrng-validation-audit-20260926.tar.zst)

The three archives total 5.42 MiB. Verify downloads against
[SHA256SUMS](https://github.com/tandem-rng/TandemRNG.jl/releases/download/statistical-evidence-2026-09-27/SHA256SUMS).

CI runs on pushes to `main`, version tags, and pull requests. Coverage reports the CPU package tests.
GPU validation and optional integration suites run separately.
