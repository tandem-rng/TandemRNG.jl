# The Tandem permutation: an asymmetric duplex on eight 32-bit words.
# The exposed half `o` is the output. The hidden half supplies the multipliers
# and masks of a Philox-shaped Feistel layer on `o`. A bijective clock (xor, rotate, one
# modular add) advances `h`, then the new first exposed word feeds back into it.
# The operations use 32-bit words:
# 32x32→64 multiply, xor, rotate, and add.
#
# The permutation is written once over a word type `W`. `UInt32` serves scalar draws, random
# access, key derivation, and the GPU kernel. `Lane8` holds the same word of the eight chunks
# of one row, so the CPU steps a whole row with one vector per state word.

const O4 = NTuple{4,UInt32}

# Domain words: distinct, nonzero, top bit set. They keep the F inputs of the
# stream, the split disciplines, and seed whitening disjoint under one key.
const DOMAIN_STREAM = 0x9e3779b9 % UInt32
const DOMAIN_SPLIT = 0xbb67ae85 % UInt32
const DOMAIN_FORK = 0xd2511f53 % UInt32
const DOMAIN_FOLD = 0xcd9e8d57 % UInt32
const DOMAIN_SEED = 0xa54ff53a % UInt32

# Aux word for stream seeding. Fork uses the child pair index instead, the others zero.
const AUX_STREAM = 0x94d049bb % UInt32

# Round constants for F, one per round, top bit set. Hex digits of 1/π.
const RC = (
    0xd17cc1b7,
    0xa7220a94,
    0xfe13abe8,
    0xfa9a6ee0,
    0xedb14acc,
    0x9e21c820,
    0xff28b1d5,
    0xef5de2b0,
    0xdb92371d,
    0xa126e970,
    0x83249775,
    0x84e8c90e,
)

# Clock parameters: rotation amounts of the xor ring and the Weyl increment.
const CLOCK_ROT = (7, 13, 22, 3)
const CLOCK_WEYL = 0x9e3779b9 % UInt32

# Number of rounds in the seeding function F.
const RF = 8

# --- the eight-lane word ------------------------------------------------------------------

# One 32-bit state word of the eight chunks of a row, so a state steps as eight vectors
# (`vector.jl`).
struct Lane8
    v::V8
end

@inline Lane8(x::UInt32) = Lane8(_vsplat(x))
@inline Lane8(t::NTuple{8,UInt32}) = Lane8(ntuple(i -> VecElement(t[i]), Val(8)))

@inline lane(a::Lane8, i::Int) = a.v[i].value

@inline Base.:⊻(a::Lane8, b::Lane8) = Lane8(_vxor(a.v, b.v))
@inline Base.:⊻(a::Lane8, b::UInt32) = Lane8(_vxor(a.v, Lane8(b).v))
@inline Base.:+(a::Lane8, b::UInt32) = Lane8(_vadd(a.v, Lane8(b).v))
@inline Base.:|(a::Lane8, b::UInt32) = Lane8(_vor(a.v, Lane8(b).v))

@inline rotl(x::UInt32, ::Val{R}) where {R} = bitrotate(x, R)
@inline rotl(a::Lane8, ::Val{R}) where {R} = Lane8(_vrotl(a.v, Val(R)))

# Lane rotations for the chain's block order (generator.jl): one lane left with a constant
# shuffle, and k lanes left with a runtime k for the rare jump path.
@inline _rotate1(a::Lane8) = Lane8(_vrotate1(a.v))
@inline _rotate(a::Lane8, k::Int) = Lane8(ntuple(i -> a.v[(i-1+k)&7+1], Val(8)))

@inline function mulwide(x::UInt32, m::UInt32)
    p = UInt64(x) * UInt64(m)
    return (p >> 32) % UInt32, p % UInt32
end

@inline function mulwide(x::Lane8, m::Lane8)
    hi, lo = _vmulwide(x.v, m.v)
    return Lane8(hi), Lane8(lo)
end

# --- the permutation ----------------------------------------------------------------------

# Four sequential single-word xor-rotates form a bijective GF(2)-linear map, and the Weyl
# add on h0 (invertible modulo 2^32) keeps the all-zero state from being a fixed point.
# The composition is a bijection, not linear or affine over either structure.
@inline function clock(h::NTuple{4,W}) where {W}
    h0, h1, h2, h3 = h
    h0 ⊻= rotl(h1, Val(CLOCK_ROT[1]))
    h1 ⊻= rotl(h2, Val(CLOCK_ROT[2]))
    h2 ⊻= rotl(h3, Val(CLOCK_ROT[3]))
    h3 ⊻= rotl(h0, Val(CLOCK_ROT[4]))
    return (h0 + CLOCK_WEYL, h1, h2, h3)
end

# One Feistel layer in the Philox4x32 shape with hidden words as multipliers and masks.
# The multiplied words a and c yield (hi, lo). The L words b and d take the fold hi ⊕ lo of
# the other pair and move to the next multiplied slots, the lo words take a clock word and
# move to the next L slots. The fold keeps the hi top-bit bias away from exposed
# words, and an odd multiplier keeps the layer a bijection of `o` for fixed `h`.
#
# Bit 0 of lo equals bit 0 of the multiplied word, and the clock word's bit 0 is
# GF(2)-linear in the hidden state unless a carry happens to rotate into it. Rotating lo by
# 16 before the key xor puts a carry-dependent product bit at bit 0 instead.
const LO_ROTATION = 16

@inline function mix(o::NTuple{4,W}, h::NTuple{4,W}) where {W}
    a, b, c, d = o
    h0, h1, h2, h3 = h
    hi0, lo0 = mulwide(a, h0 | 0x00000001)
    hi1, lo1 = mulwide(c, h1 | 0x00000001)
    return (
        b ⊻ hi1 ⊻ lo1,
        rotl(lo1, Val(LO_ROTATION)) ⊻ h2,
        d ⊻ hi0 ⊻ lo0,
        rotl(lo0, Val(LO_ROTATION)) ⊻ h3,
    )
end

# T: one step. Feedback follows the clock's addition. Undo the xor using the exposed
# output before inverting the clock and mix, so the whole step remains a bijection.
@inline function step(o::NTuple{4,W}, h::NTuple{4,W}) where {W}
    o, h = mix(o, h), clock(h)
    return o, (h[1] ⊻ o[1], h[2], h[3], h[4])
end

# F on a prepared initial state. The halves swap after every round so both get nonlinear
# mixing from a structured input, and a round constant breaks the symmetry of all-equal
# words.
@inline function seed(o::NTuple{4,W}, h::NTuple{4,W}) where {W}
    Base.Cartesian.@nexprs 12 r -> if r <= RF
        o, h = step(o, h)
        o = (o[1] ⊻ RC[r], o[2], o[3], o[4])
        o, h = h, o
    end
    return o, h
end

# F from (key, counter, domain, aux): one chunk, a child key, or a whitened seed.
@inline seed(key::O4, counter::UInt64, domain::UInt32, aux::UInt32) =
    seed((counter % UInt32, (counter >> 32) % UInt32, domain, aux), key)

# F for the eight chunks 8g + ℓ of group g at once, word-major.
@inline function seed_row(key::O4, g::UInt64)
    c0 = g << 3
    lo = Lane8(ntuple(ℓ -> (c0 + UInt64(ℓ - 1)) % UInt32, Val(8)))
    hi = Lane8(ntuple(ℓ -> ((c0 + UInt64(ℓ - 1)) >> 32) % UInt32, Val(8)))
    o = (lo, hi, Lane8(DOMAIN_STREAM), Lane8(AUX_STREAM))
    h = (Lane8(key[1]), Lane8(key[2]), Lane8(key[3]), Lane8(key[4]))
    return seed(o, h)
end
