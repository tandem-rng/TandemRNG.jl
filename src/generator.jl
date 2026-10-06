# The generator value and the bit stream law.
#
# A `Tandem8x32{K}` is an immutable value: every draw returns the value and the advanced
# generator. The stream is a bit stream. Position `pos` counts consumed bits. The stream is
# a sequence of 1024-bit rows. Row `pos >> 10` holds the 128-bit blocks of lanes 0 to 7 of
# group `row ÷ K` at step `row mod K`. Lane ℓ of group g is chunk 8g + ℓ, seeded by F, and
# its block at step j is the exposed half after j + 1 applications of T. Block `pos >> 7` is
# lane `(pos >> 7) & 7` of its row. The working state (o, h) holds the eight lanes of the row
# of `pos` after that row's step, word-major, so eight consecutive blocks cost no arithmetic
# and the next row of the group costs one eight-wide T.
#
# Lane order: `h` is in natural lane order. `o` is rotated left by the lane of `pos`, so the
# current block is always lane index 1 and a draw extracts it with constant indices. A move
# to the next block rotates `o` one lane. Eight rotations are the identity, and the eighth
# lands on the next row, so `step` always sees `o` and `h` in the same order. Indexing a
# lane at runtime would force the whole row state onto the stack on every draw.
#
# Each scalar component aligns to its width inside a block. Bool consumes one bit,
# integers their width, Float16/32/64 consume 16/32/64 bits, and Char consumes 64 bits.
# Complex values compose two real draws, so the pair can span a block boundary.

const Row = NTuple{4,Lane8}
const LANES = 8
const BLOCK_BITS = 128
const ROW_BITS = LANES * BLOCK_BITS

"""
    Tandem8x32{K}

Tandem generator with chunks of `K` steps (a power of two, default 32). Construct from an
integer seed, which is whitened through F, or from a raw 128-bit key as four `UInt32` words.
The second type parameter stores a concrete backend token. New generators use the CPU.
Bind with an MLDataDevices device, such as `rng |> CUDADevice()`, to allocate GPU arrays.

The value is immutable. `rand_next(rng, T)` returns `(x, rng′)`. `rand_fill!(rng, A)` fills
an array and returns the advanced generator. `splitrng`, `forkrng`, and `subrng` derive
child generators. Wrap in `Stateful` for the `Random` API.
"""
struct Tandem8x32{K,D<:_BackendToken}
    key::O4
    pos::UInt64
    o::Row
    h::Row
    device::D
end

Tandem8x32{K}(key::O4, pos::UInt64, o::Row, h::Row) where {K} =
    Tandem8x32{K,_CPUBackend}(key, pos, o, h, _CPUBackend())

@inline _bind(rng::Tandem8x32{K}, device::D) where {K,D<:_BackendToken} =
    Tandem8x32{K,D}(rng.key, rng.pos, rng.o, rng.h, device)

for (Device, Backend) in (
    (MLDataDevices.CPUDevice, _CPUBackend),
    (MLDataDevices.CUDADevice, _CUDABackend),
    (MLDataDevices.AMDGPUDevice, _AMDGPUBackend),
    (MLDataDevices.MetalDevice, _MetalBackend),
)
    @eval @inline (::$(Device))(rng::Tandem8x32) = _bind(rng, $Backend())
end

(device::MLDataDevices.AbstractDevice)(::Tandem8x32) =
    throw(ArgumentError("unsupported MLDataDevices device type: $(typeof(device))"))
MLDataDevices.get_device_type(rng::Tandem8x32) = MLDataDevices.get_device_type(rng.device)

@inline function _check_serviceability(
    ::Tandem8x32{K,_MetalBackend},
    ::Type{Float64},
) where {K}
    throw(ArgumentError("device execution of Float64 draws is not supported on Metal"))
end

# Copying a tuple field whole into the successor leaves LLVM a memory blob that it copies on
# every draw of a chained loop. Rebuilding the tuple from its elements keeps every word in a
# register.
@inline _by_element(words::O4) = ntuple(i -> words[i], Val(4))
@inline _by_element(row::Row) = ntuple(w -> row[w], Val(4))

@inline _rebuild(rng::Tandem8x32{K,D}, pos::UInt64, o::Row, h::Row) where {K,D} =
    Tandem8x32{K,D}(_by_element(rng.key), pos, _by_element(o), _by_element(h), rng.device)

const DEFAULT_CHUNK_LENGTH = 32
const MAX_START_POSITION = UInt64(1) << 63

@inline function _check_chunk_length(K)
    (K isa Int && K >= 1 && ispow2(K) && K <= 1 << 16) || throw(
        ArgumentError(
            "chunk length K must be an Int power of two between 1 and 65536, got $K",
        ),
    )
    return nothing
end

@inline _row(pos::UInt64) = pos >> 10
@inline _lane(pos::UInt64) = Int((pos >> 7) & 7)

# The out-of-line state builders take the key as one `UInt128`. A tuple argument travels by
# pointer in Julia's calling convention, and a pointer into the generator value would pin
# the whole value to the stack in every caller.
@inline _key128(key::O4) =
    UInt128(key[1]) | UInt128(key[2]) << 32 | UInt128(key[3]) << 64 | UInt128(key[4]) << 96
@inline _key4(k::UInt128) =
    (k % UInt32, (k >> 32) % UInt32, (k >> 64) % UInt32, (k >> 96) % UInt32)

# The row's eight lanes after its step in natural order: seed the group, then T once per
# row up to it. Out of line so the unrolled F does not sit inside every draw loop.
@noinline function _row_state(key::UInt128, row::UInt64, ::Val{K}) where {K}
    o, h = seed_row(_key4(key), row ÷ UInt64(K))
    for _ = 0:(row&UInt64(K-1))
        o, h = step(o, h)
    end
    return o, h
end

# Working state at any position: the row in natural order, then `o` rotated to the lane.
# Out of line because of the runtime lane index (see the lane order note above).
@noinline function _state_at(key::UInt128, pos::UInt64, ::Val{K}) where {K}
    o, h = _row_state(key, _row(pos), Val(K))
    ℓ = _lane(pos)
    return ntuple(w -> _rotate(o[w], ℓ), Val(4)), h
end

@inline _rotate1(o::Row) = ntuple(w -> _rotate1(o[w]), Val(4))

# The current block of a value (lane index 1 by the lane order), and the block of a given
# lane of a row in natural order for `Stateful`, whose row lives in memory where an indexed
# load costs nothing extra.
@inline _block(o::Row) = (lane(o[1], 1), lane(o[2], 1), lane(o[3], 1), lane(o[4], 1))
@inline _block(o::Row, ℓ::Int) =
    (lane(o[1], ℓ + 1), lane(o[2], ℓ + 1), lane(o[3], ℓ + 1), lane(o[4], ℓ + 1))

# The block that holds bit `p`, from its chunk alone: one F and at most K steps of T. Inlined,
# so that the callers' constant K turns the division into a shift.
@inline function _block_at(key::O4, p::UInt64, K)
    b = p >> 7
    row = b >> 3
    c = ((row ÷ UInt64(K)) << 3) | (b & 7)
    o, h = seed(key, c, DOMAIN_STREAM, AUX_STREAM)
    for _ = 0:(row&UInt64(K-1))
        o, h = step(o, h)
    end
    return o
end

"""
    Tandem8x32{K}(key::NTuple{4,UInt32}, position = 0)

Generator with the raw `key` at bit `position`. The position must be below 2^63.
"""
function Tandem8x32{K}(key::O4, pos::Integer = 0) where {K}
    _check_chunk_length(K)
    0 <= pos < MAX_START_POSITION ||
        throw(ArgumentError("start position must satisfy 0 <= position < 2^63"))
    pos = UInt64(pos)
    o, h = _state_at(_key128(key), pos, Val(K))
    return Tandem8x32{K}(key, pos, o, h)
end

Tandem8x32(key::O4, pos::Integer = 0) = Tandem8x32{DEFAULT_CHUNK_LENGTH}(key, pos)

"""
    Tandem8x32{K}(seed::Integer)

Generator whose key is `seed` whitened through the seeding function F.
`seed` must satisfy `0 <= seed < 2^128`.
"""
function Tandem8x32{K}(seed_value::Integer) where {K}
    0 <= seed_value < (BigInt(1) << 128) ||
        throw(ArgumentError("seed must satisfy 0 <= seed < 2^128"))
    s = UInt128(seed_value)
    words = (s % UInt32, (s >> 32) % UInt32, (s >> 64) % UInt32, (s >> 96) % UInt32)
    o, _ = seed(words, UInt64(0), DOMAIN_SEED, UInt32(0))
    return Tandem8x32{K}(o, 0)
end

Tandem8x32(seed_value::Integer) = Tandem8x32{DEFAULT_CHUNK_LENGTH}(seed_value)

"""
    rngkey(rng) -> NTuple{4,UInt32}

The raw key.
"""
rngkey(rng::Tandem8x32) = rng.key

"""
    rngposition(rng) -> UInt64

Bits consumed so far, including alignment gaps. Below position 2^63,
`Tandem8x32{K}(rngkey(rng), rngposition(rng))` restores the stream with CPU binding.
"""
rngposition(rng::Tandem8x32) = rng.pos

"""
    chunk_length(rng) -> Int
    chunk_length(Tandem8x32{K}) -> Int

Steps per chunk, the type parameter `K`.
"""
chunk_length(::Tandem8x32{K}) where {K} = K
chunk_length(::Type{<:Tandem8x32{K}}) where {K} = K

# Move exactly one block forward, including a row or group boundary.
@inline function _next_block(rng::Tandem8x32{K}, pos::UInt64) where {K}
    return _next_block(rng.key, rng.o, rng.h, pos, Val(K))
end

@inline function _next_block(key::O4, o::Row, h::Row, pos::UInt64, ::Val{K}) where {K}
    o = _rotate1(o)
    if _lane(pos) == 0
        row = _row(pos)
        if row & UInt64(K - 1) != 0
            o, h = step(o, h)
        else
            o, h = _row_state(_key128(key), row, Val(K))
        end
    end
    return o, h
end

# A short forward move reuses the held row and crosses at most eight block boundaries.
@inline function _advance_short(rng::Tandem8x32{K}, pos::UInt64) where {K}
    return _advance_short(
        _key128(rng.key),
        rng.pos,
        pos,
        Val(K),
        rng.o[1].v,
        rng.o[2].v,
        rng.o[3].v,
        rng.o[4].v,
        rng.h[1].v,
        rng.h[2].v,
        rng.h[3].v,
        rng.h[4].v,
    )
end

# Bare vector arguments travel by value. Passing the generator or rows by pointer would
# pin the scalar caller's whole state to the stack even when this branch is not taken.
@noinline function _advance_short(
    key::UInt128,
    from::UInt64,
    pos::UInt64,
    ::Val{K},
    o1::V8,
    o2::V8,
    o3::V8,
    o4::V8,
    h1::V8,
    h2::V8,
    h3::V8,
    h4::V8,
) where {K}
    block = from & ~UInt64(BLOCK_BITS - 1)
    target = pos & ~UInt64(BLOCK_BITS - 1)
    o = (Lane8(o1), Lane8(o2), Lane8(o3), Lane8(o4))
    h = (Lane8(h1), Lane8(h2), Lane8(h3), Lane8(h4))
    key4 = _key4(key)
    while block != target
        block += UInt64(BLOCK_BITS)
        o, h = _next_block(key4, o, h, block, Val(K))
    end
    return o, h
end

# Reuse cached state for forward moves of at most one row. Unsigned subtraction also
# handles scalar wraparound. Larger jumps and backward moves reconstruct the target.
@inline function _advance(rng::Tandem8x32{K}, pos::UInt64) where {K}
    held = rng.pos >> 7
    target = pos >> 7
    o, h = rng.o, rng.h
    if target == held + 1
        o, h = _next_block(rng, pos)
    elseif target != held
        if pos - rng.pos <= UInt64(ROW_BITS)
            o, h = _advance_short(rng, pos)
        else
            o, h = _state_at(_key128(rng.key), pos, Val(K))
        end
    end
    return _rebuild(rng, pos, o, h)
end

# --- draw widths and bit extraction --------------------------------------------------------

const FloatTypes = Union{Float16,Float32,Float64}
const WideTypes = Union{UInt128,Int128}
const ComplexTypes = Union{Complex{Float16},Complex{Float32},Complex{Float64}}
const ScalarTypes =
    Union{Bool,UInt8,Int8,UInt16,Int16,UInt32,Int32,UInt64,Int64,WideTypes,FloatTypes,Char}
const DrawTypes = Union{ScalarTypes,ComplexTypes}

@inline _check_serviceability(rng::Tandem8x32, ::Type{Complex{T}}) where {T<:FloatTypes} =
    _check_serviceability(rng, T)
@inline function _check_serviceability(
    rng::Tandem8x32{K,D},
    ::Type{T},
) where {K,D<:_GPUBackend,T<:WideTypes}
    throw(ArgumentError("$T device draws are CPU-only; move the generator to the CPU"))
end

@inline draw_size(::Type{T}) where {T<:Union{Bool,UInt8,Int8}} = 1
@inline draw_size(::Type{T}) where {T<:Union{UInt16,Int16,Float16}} = 2
@inline draw_size(::Type{T}) where {T<:Union{UInt32,Int32,Float32}} = 4
@inline draw_size(::Type{T}) where {T<:Union{UInt64,Int64,Float64,Char}} = 8
@inline draw_size(::Type{T}) where {T<:WideTypes} = 16
@inline draw_size(::Type{Complex{T}}) where {T<:FloatTypes} = 2 * draw_size(T)

# Bits one draw consumes. Bool consumes one bit, Char uses a 64-bit candidate.
@inline draw_bits(::Type{T}) where {T} = 8 * draw_size(T)
@inline draw_bits(::Type{Bool}) = 1

@inline _align_up(pos::UInt64, nbits::Int) = (pos + UInt64(nbits - 1)) & ~UInt64(nbits - 1)

# Raw bits of a draw of `s` bytes at byte offset `b` (aligned to `s`) inside the block words.
@inline _raw(o::O4, ::Int, ::Val{16}) =
    UInt128(o[1]) | (UInt128(o[2]) << 32) | (UInt128(o[3]) << 64) | (UInt128(o[4]) << 96)
@inline function _raw(o::O4, b::Int, ::Val{8})
    # Fixed choices keep the block in registers instead of spilling it for tuple indexing.
    lo = ifelse(b & 8 == 0, o[1], o[3])
    hi = ifelse(b & 8 == 0, o[2], o[4])
    return UInt64(lo) | (UInt64(hi) << 32)
end
@inline _raw(o::O4, b::Int, ::Val{4}) = o[(b>>2)+1]
@inline _raw(o::O4, b::Int, ::Val{2}) = (o[(b>>2)+1] >> (8 * (b & 3))) % UInt16
@inline _raw(o::O4, b::Int, ::Val{1}) = (o[(b>>2)+1] >> (8 * (b & 3))) % UInt8

@inline _value(::Type{Float64}, raw::UInt64) = Float64(raw >>> 11) * 0x1p-53
@inline _value(::Type{Float32}, raw::UInt32) = Float32(raw >>> 8) * Float32(0x1p-24)
@inline _value(::Type{Float16}, raw::UInt16) =
    Float16(Float32(raw >>> 5) * Float32(0x1p-11))
@inline _value(::Type{T}, raw) where {T<:Integer} = raw % T

# Match PureRNGs' fixed-work Unicode mapping, with no 128-bit device arithmetic.
@inline function _value(::Type{Char}, raw::UInt64)
    span = UInt64(0x110000 - 0x800)
    offset = ((raw >> 32) * span + (((raw & 0xffffffff) * span) >> 32)) >> 32
    code = offset % UInt32
    return Char(ifelse(code < 0xd800, code, code + UInt32(0x800)))
end

# The element of type T at bit position `p` (aligned to its width) of the block `o`.
@inline _element(::Type{T}, o::O4, p::UInt64) where {T} =
    _value(T, _raw(o, Int((p >> 3) & 15), Val(draw_size(T))))
@inline function _element(::Type{Bool}, o::O4, p::UInt64)
    b = Int(p & 127)
    return isodd(o[(b>>5)+1] >> (b & 31))
end

# The 16 bytes that the `16 ÷ s` elements of one block occupy in memory, as four words, so a
# fill can store a whole block at once. Integer elements are the stream bytes themselves and
# the floats take their bit patterns. Bool blocks expand to 128 bytes instead (fill.jl).
# Assumes a little-endian target, which every supported CPU and GPU is.
@inline _block_words(::Type{T}, o::O4) where {T<:Integer} = o
@inline _block_words(::Type{Float32}, o::O4) =
    ntuple(i -> reinterpret(UInt32, _value(Float32, o[i])), Val(4))
@inline _half_word(raw::UInt32) =
    UInt32(reinterpret(UInt16, _value(Float16, raw % UInt16))) |
    (UInt32(reinterpret(UInt16, _value(Float16, (raw >> 16) % UInt16))) << 16)
@inline _block_words(::Type{Float16}, o::O4) = ntuple(i -> _half_word(o[i]), Val(4))
@inline function _block_words(::Type{Float64}, o::O4)
    x = reinterpret(UInt64, _value(Float64, _raw(o, 0, Val(8))))
    y = reinterpret(UInt64, _value(Float64, _raw(o, 8, Val(8))))
    return (x % UInt32, (x >> 32) % UInt32, y % UInt32, (y >> 32) % UInt32)
end

"""
    rand_next(rng, T) -> (x, rng′)
    rand_next(rng, T, dims...) -> (A, rng′)

Draw one value of type `T` and return it with the advanced generator. `T` is `Bool`,
an 8- to 128-bit integer, `Float16`, `Float32`, `Float64`, a complex of those floats,
or `Char`. Real floats lie in [0, 1); complex values compose two real draws.
Char uses the fixed-work Unicode scalar mapping specified in the Tandem8x32 specification.
Array draws allocate on the generator's backend. Scalar calls run where they are called:
on the host, or inside a device kernel. Binding preserves the key and position.
"""
@inline function rand_next(rng::Tandem8x32{K}, ::Type{T}) where {K,T<:ScalarTypes}
    nb = draw_bits(T)
    # A Bool is always aligned. Its chain runs about a fifth faster on the general path.
    nb == 1 && return _rand_next_general(rng, T)
    pos = rng.pos
    after = pos + UInt64(nb)
    # Most draws start aligned and end inside the block, so they read the held block and
    # change no state. One non-short-circuit test sends the others to the general path.
    # With a separate alignment test, LLVM laid out the aligned case as a taken branch with
    # state copies, and the 8- to 32-bit chains ran about 30 % slower.
    if (pos & UInt64(nb - 1) != 0) | (after & UInt64(BLOCK_BITS - 1) == 0)
        return _rand_next_general(rng, T)
    end
    return _element(T, _block(rng.o), pos), _rebuild(rng, after, rng.o, rng.h)
end

@inline function _rand_next_general(rng::Tandem8x32{K}, ::Type{T}) where {K,T}
    nb = draw_bits(T)
    pos = rng.pos
    # Rounding up runs only when the previous draw left the position misaligned, which
    # happens once after a change of draw type. It may also push into the next block.
    if pos & UInt64(nb - 1) != 0
        pos = _align_up(pos, nb)
        rng = _advance(rng, pos)
    end
    x = _element(T, _block(rng.o), pos)
    pos += UInt64(nb)
    # An aligned scalar draw can only stay in its block or reach the next block's start.
    o, h = rng.o, rng.h
    if pos & UInt64(BLOCK_BITS - 1) == 0
        o, h = _next_block(rng, pos)
    end
    return x, _rebuild(rng, pos, o, h)
end

@inline function rand_next(rng::Tandem8x32, ::Type{Complex{T}}) where {T<:FloatTypes}
    re, rng = rand_next(rng, T)
    im, rng = rand_next(rng, T)
    return Complex{T}(re, im), rng
end

"""
    rand_at(rng, T, i) -> x

The `i`-th value (1-based) of `rand_fill!(rng, Vector{T}(undef, n))` without advancing
`rng`. Each real component costs one F plus at most `K` steps of T. Use this for
addressed draws; use `rand_next` or `rand_fill!` for sequential draws.
"""
function rand_at(rng::Tandem8x32{K}, ::Type{T}, i::Integer) where {K,T<:ScalarTypes}
    i >= 1 || throw(ArgumentError("index must be at least 1"))
    nb = draw_bits(T)
    p = _align_up(rng.pos, nb) + UInt64(nb) * (UInt64(i) - 1)
    return _element(T, _block_at(rng.key, p, K), p)
end

function rand_at(rng::Tandem8x32, ::Type{Complex{T}}, i::Integer) where {T<:FloatTypes}
    i >= 1 || throw(ArgumentError("index must be at least 1"))
    j = 2 * (UInt64(i) - 1)
    return Complex{T}(rand_at(rng, T, j + 1), rand_at(rng, T, j + 2))
end
