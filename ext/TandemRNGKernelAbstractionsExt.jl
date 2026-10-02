# Portable GPU fill. One work-item per chunk: seed with F,
# run K steps of T, store each block where the stream law puts it. Lane ℓ of a group is chunk
# 8g + ℓ, and the eight lanes of a group write the eight adjacent blocks of a row at every
# step, so adjacent work-items store adjacent 16-byte blocks and a warp fills whole lines.
# For K >= 8 the rows of a workgroup are staged in shared memory so a warp writes 512
# contiguous bytes (`fill_tile!`); smaller K uses direct stores (`fill_lanes!`). The law is
# the CPU law bit for bit, so a device fill equals a CPU fill of the same range.
#
# A Bool fill runs in two passes: the word fill above into a temporary array one bit per
# Bool, then an expansion kernel in which adjacent work-items write adjacent 16-byte pieces.
# Expanding inside the chunk kernel put each work-item's 128 Bool bytes 128 bytes from its
# neighbour's and ran at 350 GiB/s on an A100 against 1300 for the other types.
module TandemRNGKernelAbstractionsExt

using TandemRNG
using TandemRNG:
    O4,
    V4,
    DOMAIN_STREAM,
    AUX_STREAM,
    BLOCK_BITS,
    ROW_BITS,
    DrawTypes,
    draw_bits,
    _align_up,
    _element,
    _block_words,
    _store_block!,
    _v4,
    _advance,
    seed,
    step
using KernelAbstractions
using GPUArraysCore: AbstractGPUArray

const WORKGROUP = 256

# Device arrays hand out `Core.LLVMPtr`, whose store takes the alignment that makes the
# backend emit one 16-byte vector store.
@inline TandemRNG._store_block!(p::Core.LLVMPtr{T,AS}, w::O4) where {T,AS} =
    unsafe_store!(reinterpret(Core.LLVMPtr{V4,AS}, p), _v4(w), 1, Val(16))

# Blocks start at multiples of 128 bits in the stream. Their 16 bytes are 16-byte aligned in
# the array when the array's first byte and the fill's first stream bit agree modulo a
# block. `pointer(A)` is a device address, but only its residue matters here.
_blocks_aligned(::AbstractArray, ::UInt64) = false
_blocks_aligned(A::AbstractGPUArray, p0::UInt64) = (8 * UInt(pointer(A)) - p0) % 128 == 0
_blocks_aligned(::AbstractGPUArray{Char}, ::UInt64) = false

# Store the block whose first stream bit is `b` where it meets [p0, pend): one vector store
# when the whole block lies inside and the array's blocks are aligned, element stores
# otherwise.
@inline function _store_in_range!(
    A,
    o::O4,
    b::UInt64,
    p0::UInt64,
    pend::UInt64,
    ::Val{ALIGNED},
) where {ALIGNED}
    T = eltype(A)
    nb = draw_bits(T)
    per_block = BLOCK_BITS ÷ nb
    if p0 <= b && UInt64(BLOCK_BITS) <= pend - b
        first = Int((b - p0) ÷ UInt64(nb)) + 1
        if ALIGNED
            _store_block!(pointer(A, first), _block_words(T, o))
        else
            for w = 0:(per_block-1)
                @inbounds A[first+w] = _element(T, o, b + UInt64(nb * w))
            end
        end
    elseif p0 <= b || p0 - b < UInt64(BLOCK_BITS)
        for w = 0:(per_block-1)
            bit = b + UInt64(nb * w)
            if p0 <= bit && bit < pend
                @inbounds A[Int((bit-p0)÷UInt64(nb))+1] = _element(T, o, bit)
            end
        end
    end
    return nothing
end

# One work-item per chunk, direct stores. Any K.
@kernel function fill_lanes!(
    A,
    key::O4,
    c0::UInt64,
    p0::UInt64,
    pend::UInt64,
    ::Val{K},
    ::Val{ALIGNED},
) where {K,ALIGNED}
    c = c0 + UInt64(@index(Global, Linear) - 1)
    o, h = seed(key, c, DOMAIN_STREAM, AUX_STREAM)
    # First block of this chunk: row K·g of group g = c >> 3, lane c & 7, in bits.
    b = (((c >> 3) * UInt64(K)) << 10) | ((c & 7) << 7)
    for _ = 1:K
        b >= pend && break
        o, h = step(o, h)
        _store_in_range!(A, o, b, p0, pend, Val(ALIGNED))
        b += ROW_BITS
    end
end

# A direct store covers one 128-byte row per warp-wide step. Staging TILE_STEPS rows of each
# group in shared memory lets consecutive work-items write consecutive 16-byte slots, so a
# warp writes 512 contiguous bytes. On an A100 this raised the fill from 1309 to 1395 GiB/s
# in tandem-cuda, whose kernel this mirrors.
const TILE_STEPS = 8
const TILE_GROUPS = WORKGROUP ÷ 8
const TILE_SLOTS = TILE_GROUPS * TILE_STEPS * 8

# One work-item per chunk, TILE_GROUPS groups per workgroup, output staged through shared
# memory. Needs K >= TILE_STEPS. `g0` is the first group of the fill, `g1` the last.
@kernel function fill_tile!(
    A,
    key::O4,
    g0::UInt64,
    g1::UInt64,
    p0::UInt64,
    pend::UInt64,
    ::Val{K},
    ::Val{ALIGNED},
) where {K,ALIGNED}
    tile = @localmem V4 (TILE_SLOTS,)
    t = @index(Local, Linear) - 1
    gb = g0 + UInt64(@index(Group, Linear) - 1) * UInt64(TILE_GROUPS)
    gi = t >> 3
    lane = t & 7
    g = gb + UInt64(gi)
    mine = g <= g1
    o, h = seed(key, (g << 3) | UInt64(lane), DOMAIN_STREAM, AUX_STREAM)
    # Stream bit of the workgroup's first row, then of the tile's first row.
    block_first = (gb * UInt64(K)) << 10
    for jb = 0:TILE_STEPS:(K-1)
        tile_first = block_first + UInt64(jb) * ROW_BITS
        tile_first >= pend && break
        if mine
            for j = 0:(TILE_STEPS-1)
                o, h = step(o, h)
                @inbounds tile[(gi*TILE_STEPS+j)*8+lane+1] = _v4(o)
            end
        end
        @synchronize
        for s = t:WORKGROUP:(TILE_SLOTS-1)
            sg = s ÷ (TILE_STEPS * 8)
            within = s % (TILE_STEPS * 8)
            b =
                (((gb + UInt64(sg)) * UInt64(K) + UInt64(jb)) << 10) +
                UInt64(within) * UInt64(BLOCK_BITS)
            b >= pend && continue
            @inbounds v = tile[s+1]
            _store_in_range!(
                A,
                (v[1].value, v[2].value, v[3].value, v[4].value),
                b,
                p0,
                pend,
                Val(ALIGNED),
            )
        end
        @synchronize
    end
end

function TandemRNG.rand_fill!(
    rng::Tandem8x32{K,D},
    A::AbstractArray{T};
    nthreads::Integer = 1,
) where {K,D<:TandemRNG._GPUBackend,T<:TandemRNG.ScalarTypes}
    nthreads >= 1 || throw(ArgumentError("nthreads must be at least 1"))
    TandemRNG._check_fill_device(rng, A)
    TandemRNG._check_serviceability(rng, T)
    nb = draw_bits(T)
    n = UInt64(length(A))
    p0 = _align_up(rng.pos, nb)
    p0 >= rng.pos && n <= (typemax(UInt64) - p0) ÷ UInt64(nb) ||
        throw(ArgumentError("fill of $(length(A)) elements passes the end of the stream"))
    n == 0 && return _advance(rng, p0)
    pend = p0 + UInt64(nb) * n
    _fill_bits!(rng, A, p0, pend)
    return _advance(rng, pend)
end

# Queue all words intersecting [p0, pend). Bool staging may need only part of its last
# UInt32, including the final word of the stream, whose rounded end would overflow.
function _fill_bits!(rng::Tandem8x32{K}, A, p0::UInt64, pend::UInt64) where {K}
    bits_per_group = UInt64(ROW_BITS) * UInt64(K)
    g0 = p0 ÷ bits_per_group
    g1 = (pend - 1) ÷ bits_per_group
    backend = get_backend(A)
    aligned = Val(_blocks_aligned(A, p0))
    if K >= TILE_STEPS
        ngroups = Int(g1 - g0 + 1)
        kernel = fill_tile!(backend, WORKGROUP)
        kernel(
            A,
            rng.key,
            g0,
            g1,
            p0,
            pend,
            Val(K),
            aligned;
            ndrange = cld(ngroups, TILE_GROUPS) * WORKGROUP,
        )
    else
        kernel = fill_lanes!(backend, WORKGROUP)
        kernel(
            A,
            rng.key,
            g0 << 3,
            p0,
            pend,
            Val(K),
            aligned;
            ndrange = Int(g1 - g0 + 1) * 8,
        )
    end
    # No host synchronize: the fill is ordered on the device queue like any other array
    # operation, and the wait cost 2 to 7 % of the fill's throughput on an A100.
    return nothing
end

# --- Bool: word fill, then expansion -------------------------------------------------------

# Four bits of `x` (after the shift) as four bytes of 0 or 1. The multiply places the bits
# at distances of eight without carries between them.
@inline _spread4(x::UInt32, shift::Int) = ((x >> shift) & 0x0f) * 0x00204081 & 0x01010101

# Work-item t writes Bools 16(t − 1) to 16t − 1 from bits `off + 16(t − 1)` of the words `W`.
@kernel function expand_bools!(A, W, off::Int, n::Int, ::Val{ALIGNED}) where {ALIGNED}
    i0 = 16 * (@index(Global, Linear) - 1)
    bit = off + i0
    w = bit >> 5
    sh = bit & 31
    @inbounds lo = UInt64(W[w+1])
    hi = sh > 16 && n - i0 > 32 - sh ? UInt64(@inbounds W[w+2]) : UInt64(0)
    x = ((lo | hi << 32) >> sh) % UInt32
    if i0 + 16 <= n
        if ALIGNED
            _store_block!(
                pointer(A, i0 + 1),
                (_spread4(x, 0), _spread4(x, 4), _spread4(x, 8), _spread4(x, 12)),
            )
        else
            for j = 0:15
                @inbounds A[i0+1+j] = isodd(x >> j)
            end
        end
    else
        for j = 0:(n-i0-1)
            @inbounds A[i0+1+j] = isodd(x >> j)
        end
    end
end

function TandemRNG.rand_fill!(
    rng::Tandem8x32{K,D},
    A::AbstractArray{Bool};
    nthreads::Integer = 1,
) where {K,D<:TandemRNG._GPUBackend}
    nthreads >= 1 || throw(ArgumentError("nthreads must be at least 1"))
    TandemRNG._check_fill_device(rng, A)
    n = length(A)
    p0 = rng.pos
    UInt64(n) <= typemax(UInt64) - p0 ||
        throw(ArgumentError("fill of $n elements passes the end of the stream"))
    n == 0 && return rng
    pend = p0 + UInt64(n)
    # Words from the 32-bit boundary at or before p0, so the word fill is aligned and the
    # expansion skips the first `off` bits.
    pw = p0 & ~UInt64(31)
    off = Int(p0 - pw)
    nwords = Int(cld(pend - pw, UInt64(32)))
    backend = get_backend(A)
    W = KernelAbstractions.allocate(backend, UInt32, nwords)
    _fill_bits!(rng, W, pw, pend)
    aligned = _blocks_aligned(A, UInt64(0))
    expand_bools!(backend, WORKGROUP)(A, W, off, n, Val(aligned); ndrange = cld(n, 16))
    # A no-op on backends without an early free (KernelAbstractions 0.9 with CUDA), where
    # the collector reclaims the words. The launch is queued, so the release is ordered.
    KernelAbstractions.unsafe_free!(W)
    return _advance(rng, pend)
end

end
