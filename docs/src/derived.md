# Derived draws

Bounded integers, normals, and exponentials follow Appendix A of the specification.
Every Tandem port computes them the same way, so TandemRNG returns the values of
tandem-c and tandem-cuda bit for bit. The test suite checks this against their fixtures.

```@example derived
using TandemRNG, Random
rng = Tandem8x32(42)
die, rng = rand_next(rng, 1:6)
z, rng = normal_next(rng)
e, rng = exponential_next(rng, Float32)
A = Vector{Float64}(undef, 1000)
rng = normal_fill!(rng, A)
st = Stateful(rng)
(die, z, e, rand(st, 1:6), randn(st), randexp(st))
```

## Bounded integers

[`rand_next`](@ref) and [`rand_fill!`](@ref) take an integer range. `Stateful` gives
the same values through `rand(st, r)`, `rand!(st, A, r)`, and every sampler that picks
an index range, such as `rand(st, collection)`.

- Draws use Lemire's multiply-and-reject method on uniform `w`-bit draws.
- The width comes from the number of values `m = length(r)`: `w = 32` when `m ≤ 2^32`,
  else `w = 64`. The element type of the range does not change the value.
- `lo:hi` draws on `m = hi − lo + 1` and adds `lo`.
- A scalar draw rejects by drawing the next `w` bits.
- Element `i` of a fill maps draw `i` of the plain `w`-bit fill, so a fill consumes
  exactly one draw per element. A rejected draw retries on a fallback generator:
  `splitrng` index `g` of `subrng` purpose `P_w` of the key at position 0.
  `g` is the draw's index in the stream, the aligned start position over `w` plus `i`.
  `P_32 = 0x424c573332` and `P_64 = 0x424c573634` are reserved for this.
- A fill cut at any element boundary therefore equals the whole fill.
  A fill without rejections equals the sequence of scalar draws.
- An empty fill leaves the position unchanged.

[`rand_below_next`](@ref) and [`rand_below_fill!`](@ref) name the width by the type of
`n`, as tandem-c's `tandem_u32_below` and `tandem_fill_u64_below` do. `n = 0` returns 0
after one draw. Immutable range draws reject ranges of more than 2^64 values.
`Stateful` keeps Random's sampler for those 128-bit ranges.

## Normals

### Float64: ziggurat

Float64 normals use the 1024-layer ziggurat of Appendix A, one UInt64 draw per element.

- Element `i` of a fill comes from UInt64 draw `i`, and a fill of `n` elements consumes `n`
  draws. A scalar draw consumes one draw and equals element 1 of a fill. An empty fill
  aligns the position to 64 bits.
- 99.57 % of the draws land in an inner rectangle and cost a table lookup and a multiply.
- A draw that misses continues on its own fallback generator: `splitrng` index `g` of
  `subrng` purpose `0x4e524d3634` of the key at position 0, where `g` is the draw's index in
  the stream. So a fill cut at any element equals the whole fill, and threads and workers
  may start anywhere.
- The tables come from the spec's `tables/normal_f64_zig1024.json`. `tools/gen_zig_tables.jl`
  writes them into `src/zig_tables.jl` after it checks the file's SHA-256, and CI checks the
  generated file is current. The logarithm is tandem-c's polynomial, so the values are exact
  across ports.

### Float32: Box-Muller

Float32 normals use the Box-Muller transform on two Float32 uniform draws `a` and `b`:

```
r  = sqrt(−2 log(1 − a))
z₀ = r cos(2π b)
z₁ = r sin(2π b)
```

- Fill elements `2j − 1` and `2j` are `z₀` and `z₁` of uniform draws `2j − 1` and `2j`.
- A fill of `n` elements consumes `2·cld(n, 2)` uniforms. An odd `n` writes only `z₀` of
  its last pair. An empty fill leaves the position unchanged.
- A scalar draw returns `z₀` and consumes two uniforms, so it equals element 1 of a fill.
  `randn(st, Float32)` on `Stateful` does the same and keeps no `z₁`.
- They are computed in Float32.
- `log`, `cos`, and `sin` are tandem-c's short polynomials with explicit fused
  multiply-adds and no library call. Julia does not contract other products into sums,
  so the bits match on every CPU.

To split a Float32 normal fill across workers, start each range at an odd element (an even
offset), so that the pairs fall the same way. Only the last range may have odd length.

## Exponentials

[`exponential_next`](@ref) and [`exponential_fill!`](@ref) return `−log(1 − u)` for one
uniform draw `u` of the output type. Float64 uses the polynomial `log` of the normals.
Float32 carries the leading term in two floats and adds `k log 2` by an exact two-sum, within
0.58 ulp, as tandem-c does. A fill consumes one uniform per element, and element 1 equals the
scalar draw.

## Weighted choice

[`ChoiceTable`](@ref) builds Walker's alias table of Appendix C from the weights in exact
integers, so the table equals tandem-c's. [`choice_next`](@ref) and [`choice_fill!`](@ref)
draw one-based indices, one UInt64 draw per element, and an empty fill aligns the position
to 64 bits. `Stateful` gives the same values through `rand(st, t)` and `rand!(st, A, t)`.

## Threads and devices

Normal and exponential fills split their groups over `nthreads` tasks as `rand_fill!`
does, and the values do not depend on the thread count. Bounded fills thread the uniform
fill and map the draws in one task. Derived fills run on the CPU only.

The PureRNGs extension keeps PureRNGs' own samplers: `PureRNGs.randn_next` and the other
PureRNGs distribution draws do not follow Appendix A.

## Tests

`test/derived.jl` checks the values against copies of tandem-c's `cross_below.h` and
`cross_normal.h`, tandem-cuda's `cross_fill_below.h` and `cross_fill_exponential.h`, the
latter generated from tandem-cuda e98daee's core, the spec's `conformance/exponential.json`, and the SHA-256 of
tandem-c's 1e6-element normal dump. The exponential dump matches the SHA-256 and FNV-1a of
the spec's `conformance/hashes.json`. `test/metal` checks Float32 exponentials drawn in Metal
kernels against the CPU fill. The `cross_normal.h` rows put a wedge accept, a wedge reject and a tail
at unaligned starts. The normals also match the FNV-1a hashes of tandem-c's
`test_normal_bits.c`, including the spec's Python implementation of Appendix A. Normals and
exponentials pass moment and Kolmogorov-Smirnov tests on 1e7 draws. `test/choice.jl`
checks the weighted choice tables and indices against a copy of the spec's
`conformance/choice.json`, plus a chi-square test on 1e6 draws.
