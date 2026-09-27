module TandemRNGPureRNGsKernelAbstractionsExt

import TandemRNG as TR
import PureRNGs as PR
using KernelAbstractions

# Device cursors use scalar words, so no host UInt128 state-builder argument
# enters Metal code. Permutations and child-key derivation remain Tandem's own.
struct _TileEngine{K,D}
    key::TR.O4
    device::D
end

@inline PR._engine_backend(rng::_TileEngine) = rng.device

# The rare Gamma fallback usually reads only one block. Carrying eight rows
# here increases every sampler kernel's register and local-memory requirements.
struct _FallbackRNG{K}
    key::TR.O4
    pos::UInt64
    o::TR.O4
end

@inline function PR._child_cursor(rng::_TileEngine{K}, purpose::UInt64) where {K}
    key = TR._child_key(rng.key, purpose, TR.DOMAIN_FOLD, UInt32(0), 0)
    child = _FallbackRNG{K}(key, UInt64(0), TR._block_at(key, UInt64(0), K))
    return child, child
end

@inline function _advance_child(rng::_FallbackRNG{K}, pos::UInt64) where {K}
    o = (pos >> 7) == (rng.pos >> 7) ? rng.o : TR._block_at(rng.key, pos, K)
    return _FallbackRNG{K}(rng.key, pos, o)
end

@inline function PR._take_bits(::_FallbackRNG, cursor::_FallbackRNG, ::Val{W}) where {W}
    slot = W == 1 ? 1 : max(8, nextpow(2, W))
    cursor = _advance_child(cursor, TR._align_up(cursor.pos, slot))
    raw =
        slot == 1 ? UInt64(TR._element(Bool, cursor.o, cursor.pos)) :
        UInt64(TR._raw(cursor.o, Int((cursor.pos >> 3) & 15), Val(slot ÷ 8)))
    return raw >> (slot - W), _advance_child(cursor, cursor.pos + UInt64(slot))
end

@inline function _store_draw!(destination, index, codec, rng, cursor)
    value, _ = PR._codec_take(codec, rng, cursor, eltype(destination))
    @inbounds destination[index] = value
    return nothing
end

# The log-gamma matrix in column-major order is one component span per element.
@inline function _store_draw!(destination, index, codec::PR._DirichletCodec, rng, cursor)
    component = (index - 1) % length(codec.alpha) + 1
    @inbounds destination[index] =
        PR._component_log(codec, rng, cursor, component, eltype(destination))
    return nothing
end

@kernel function _normalize_columns!(destination)
    column = @index(Global, Linear)
    PR._normalize_column!(destination, column)
end

# The cursor reads a tile in native little-endian word order. Its absolute bit
# position also keys the Gamma fallback, independently of the tile boundaries.
struct _TileCursor{A,S}
    words::A
    pos::UInt64
    origin::UInt64
end

@inline _TileCursor(words, pos, origin, ::Val{S}) where {S} =
    _TileCursor{typeof(words),S}(words, pos, origin)

@inline function PR._take_bits(
    ::_TileEngine,
    cursor::_TileCursor{A,S},
    ::Val{W},
) where {A,S,W}
    if eltype(cursor.words) === UInt64
        index = Int((cursor.pos - cursor.origin) >> 6) + 1
        raw = @inbounds cursor.words[index]
        S == 64 || (raw = (raw >> Int(cursor.pos & 63)) & ((UInt64(1) << S) - UInt64(1)))
    else
        index = Int((cursor.pos - cursor.origin) >> 5) + 1
        raw = UInt64(@inbounds cursor.words[index])
        raw = (raw >> Int(cursor.pos & 31)) & ((UInt64(1) << S) - UInt64(1))
    end
    next = _TileCursor(cursor.words, cursor.pos + UInt64(S), cursor.origin, Val(S))
    return raw >> (S - W), next
end

@inline PR._skip_takes(
    ::_TileEngine,
    cursor::_TileCursor{A,S},
    count::Integer,
    ::Val,
) where {A,S} =
    _TileCursor(cursor.words, cursor.pos + UInt64(count) * UInt64(S), cursor.origin, Val(S))
@inline PR._cursor_ordinal(::_TileEngine, cursor::_TileCursor) = cursor.pos

# Tiles follow native stream boundaries. A draw belongs to the tile containing
# its first slot, and a short overlap holds its remaining candidates. This
# avoids rebuilding the prefix of a chunk group at every sampler boundary.
@kernel function _fill_tile!(
    destination,
    rng::_TileEngine{K},
    codec,
    start::UInt64,
    stop::UInt64,
    ::Val{C},
    ::Val{S},
    ::Val{B},
    ::Val{L},
) where {K,C,S,B,L}
    thread = @index(Local, Linear)
    tile = @index(Group, Linear) - 1
    origin = (start ÷ UInt64(B)) * UInt64(B) + UInt64(tile) * UInt64(B)
    edge = origin + min(UInt64(B), stop - origin)
    first = origin <= start ? 0 : Int(cld(origin - start, UInt64(C * S)))
    last = min(length(destination), Int(cld(edge - start, UInt64(C * S))))
    p0 = start + UInt64(first) * UInt64(C * S)
    pend = start + UInt64(last) * UInt64(C * S)
    @uniform word_bits = S <= 32 ? 32 : 64
    words = @localmem (S <= 32 ? UInt32 : UInt64) (B ÷ word_bits + cld(C * S, word_bits))
    group_bits = UInt64(K * TR.ROW_BITS)
    g0 = p0 ÷ group_bits
    g1 = (pend - UInt64(1)) ÷ group_bits
    chunks = Int(g1 - g0 + 1) * TR.LANES
    for offset = (thread-1):L:(chunks-1)
        chunk = (g0 << 3) + UInt64(offset)
        bit = (((chunk >> 3) * UInt64(K)) << 10) | ((chunk & 7) << 7)
        if bit < pend
            o, h = TR.seed(rng.key, chunk, TR.DOMAIN_STREAM, TR.AUX_STREAM)
            if origin <= bit && UInt64((K - 1) * TR.ROW_BITS + TR.BLOCK_BITS) <= pend - bit
                for _ = 1:K
                    o, h = TR.step(o, h)
                    TR._store_block!(
                        pointer(words, Int((bit-origin)÷UInt64(word_bits))+1),
                        o,
                    )
                    bit += UInt64(TR.ROW_BITS)
                end
            else
                for _ = 1:K
                    bit >= pend && break
                    o, h = TR.step(o, h)
                    for word = 1:(TR.BLOCK_BITS÷word_bits)
                        position = bit + UInt64(word_bits * (word - 1))
                        if origin <= position < pend
                            raw =
                                word_bits == 32 ? o[word] :
                                UInt64(o[2word-1]) | (UInt64(o[2word]) << 32)
                            @inbounds words[Int((position-origin)÷UInt64(word_bits))+1] =
                                raw
                        end
                    end
                    bit += UInt64(TR.ROW_BITS)
                end
            end
        end
    end
    @synchronize
    for draw = (first+thread):L:last
        position = start + UInt64(draw - 1) * UInt64(C * S)
        cursor = _TileCursor(words, position, origin, Val(S))
        _store_draw!(destination, draw, codec, rng, cursor)
    end
end

TR._fill_mapped!(
    rng::TR.Tandem8x32{K,D},
    destination,
    codec,
    count::Integer,
    width::Val,
    slot::Val,
) where {K,D<:TR._GPUBackend} = _fill_tiles!(rng, destination, codec, count, slot)

function _fill_tiles!(
    rng::TR.Tandem8x32{K},
    destination,
    codec,
    count::Integer,
    slot::Val{S},
) where {K,S}
    start = TR._align_up(rng.pos, S)
    backend = PR._engine_backend(rng)
    engine = _TileEngine{K,typeof(backend)}(rng.key, backend)
    # A 16 KiB input tile leaves 16 KiB for a draw crossing its end.
    bits = 131072
    stop = start + UInt64(length(destination)) * UInt64(count * S)
    threads = count <= 2 ? (S == 32 ? 128 : min(256, 8192 ÷ S)) : 64
    groups = Int(cld(stop - (start ÷ UInt64(bits)) * UInt64(bits), UInt64(bits)))
    kernel = _fill_tile!(get_backend(destination), threads)
    kernel(
        destination,
        engine,
        codec,
        start,
        stop,
        Val(count),
        slot,
        Val(bits),
        Val(threads);
        ndrange = groups * threads,
    )
    return nothing
end

# A column fill runs in two phases: the log-gamma matrix as an element fill of
# the component spans, so a workgroup holds a workitem per component instead of
# one per column, then a workitem per column for the normalization.
function TR._fill_mapped!(
    rng::TR.Tandem8x32{K,D},
    destination,
    codec::PR._DirichletCodec,
    count::Integer,
    width::Val,
    slot::Val,
) where {K,D<:TR._GPUBackend}
    component_count, _ = PR._component_takes(codec, eltype(destination))
    _fill_tiles!(rng, destination, codec, component_count, slot)
    normalize! = _normalize_columns!(get_backend(destination), 256)
    normalize!(destination; ndrange = size(destination, 2))
    return nothing
end

end
