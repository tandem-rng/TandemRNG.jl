# A bulk cursor keeps the row in natural lane order. It changes the row state
# only at a row boundary, including skips over unread rejection candidates.
struct _RowCursor
    pos::UInt64
    o::TR.Row
    h::TR.Row
end

@inline function _row_cursor(rng::TR.Tandem8x32)
    lane = TR._lane(rng.pos)
    return _RowCursor(rng.pos, ntuple(w -> TR._rotate(rng.o[w], -lane), Val(4)), rng.h)
end

@inline function _advance_row(
    rng::TR.Tandem8x32{K},
    cursor::_RowCursor,
    pos::UInt64,
) where {K}
    o, h = cursor.o, cursor.h
    row_start = cursor.pos & ~UInt64(TR.ROW_BITS - 1)
    target = pos & ~UInt64(TR.ROW_BITS - 1)
    while row_start != target
        row_start += UInt64(TR.ROW_BITS)
        row = TR._row(row_start)
        if iszero(row & UInt64(K - 1))
            o, h = TR.seed_row(rng.key, row ÷ UInt64(K))
        end
        o, h = TR.step(o, h)
    end
    return _RowCursor(pos, o, h)
end

@inline function _slot_raw(o::TR.O4, pos::UInt64, width::Val{W}) where {W}
    slot = _slot_bits(width)
    raw = if slot <= 32
        # Fixed selections keep the four words in registers during cursor reads.
        low = ifelse(iszero(pos & 32), o[1], o[2])
        high = ifelse(iszero(pos & 32), o[3], o[4])
        word = ifelse(iszero(pos & 64), low, high)
        slot == 32 ? UInt64(word) :
        UInt64((word >> Int(pos & 31)) & UInt32((UInt64(1) << slot) - 1))
    else
        UInt64(TR._raw(o, Int((pos >> 3) & 15), Val(slot ÷ 8)))
    end
    return raw >> (slot - W)
end

@inline function PureRNGs._take_bits(rng::TR.Tandem8x32, cursor::_RowCursor, width::Val)
    raw = _slot_raw(TR._block(cursor.o, TR._lane(cursor.pos)), cursor.pos, width)
    return raw, _advance_row(rng, cursor, cursor.pos + UInt64(_slot_bits(width)))
end

@inline PureRNGs._skip_takes(
    rng::TR.Tandem8x32,
    cursor::_RowCursor,
    count::Integer,
    width::Val,
) = _advance_row(rng, cursor, cursor.pos + UInt64(count) * UInt64(_slot_bits(width)))
@inline PureRNGs._cursor_ordinal(::TR.Tandem8x32, cursor::_RowCursor) = cursor.pos

# This write-only view lets the native fill own seeding, row generation,
# partial rows, and thread partitioning. Only its stores apply the codec.
struct _MappedArray{T,N,A,C,W} <: AbstractArray{T,N}
    destination::A
    codec::C
end

_MappedArray(destination::AbstractArray{T,N}, codec::C, width::Val{W}) where {T,N,C,W} =
    _MappedArray{_slot_type(width),N,typeof(destination),C,W}(destination, codec)
Base.size(array::_MappedArray) = size(array.destination)
Base.IndexStyle(::Type{<:_MappedArray}) = IndexLinear()
TR.MLDataDevices.get_device_type(array::_MappedArray) =
    TR.MLDataDevices.get_device_type(array.destination)

@inline function Base.setindex!(
    array::_MappedArray{T,N,A,C,W},
    value,
    index::Int,
) where {T,N,A,C,W}
    raw = UInt64(value) >> (_slot_bits(Val(W)) - W)
    @inbounds array.destination[index] =
        PureRNGs._cooperative_value(array.codec, eltype(array.destination), raw)
    return array
end

@inline function _row_raws(o::TR.Row, width::Val)
    slot = _slot_bits(width)
    return ntuple(Val(TR.ROW_BITS ÷ slot)) do i
        bit = UInt64((i - 1) * slot)
        _slot_raw(TR._block(o, TR._lane(bit)), bit, width)
    end
end

@inline function TR._write_row!(
    array::_MappedArray{T,N,A,C,W},
    o::TR.Row,
    row::UInt64,
    start::UInt64,
) where {T,N,A,C,W}
    raws = _row_raws(o, Val(W))
    first = Int((row - start) ÷ UInt64(_slot_bits(Val(W)))) + 1
    # Tuple indexing loses the word's range information. Restating its width
    # lets Julia use the bounded integer-to-float conversion in the codec.
    mask = W == 64 ? typemax(UInt64) : (UInt64(1) << W) - UInt64(1)
    @inbounds for i in eachindex(raws)
        array.destination[first+i-1] = PureRNGs._cooperative_value(
            array.codec,
            eltype(array.destination),
            raws[i] & mask,
        )
    end
    return nothing
end
