# PureRNGs owns the codecs. Tandem owns their aligned slots and continuation state.
for Backend in (:_CPUBackend, :_CUDABackend, :_AMDGPUBackend, :_MetalBackend)
    @eval @inline PureRNGs._engine_backend(rng::TR.Tandem8x32{K,TR.$Backend}) where {K} =
        PureRNGs.$Backend()
end

@inline _slot_bits(::Val{W}) where {W} = W == 1 ? 1 : max(8, nextpow(2, W))
@inline _slot_type(::Val{W}) where {W} =
    W == 1 ? Bool : W <= 8 ? UInt8 : W <= 16 ? UInt16 : W <= 32 ? UInt32 : UInt64

@inline function PureRNGs._take_bits(
    ::TR.Tandem8x32,
    cursor::TR.Tandem8x32,
    width::Val{W},
) where {W}
    raw, cursor = TR.rand_next(cursor, _slot_type(width))
    return UInt64(raw) >> (_slot_bits(width) - W), cursor
end

# A rejection sampler skips its unread candidates without reconstructing each row.
@inline function PureRNGs._skip_takes(
    ::TR.Tandem8x32,
    cursor::TR.Tandem8x32,
    count::Integer,
    width::Val,
)
    bits = UInt64(count) * UInt64(_slot_bits(width))
    while bits > UInt64(TR.ROW_BITS)
        cursor = TR._advance(cursor, cursor.pos + UInt64(TR.ROW_BITS))
        bits -= UInt64(TR.ROW_BITS)
    end
    return TR._advance(cursor, cursor.pos + bits)
end

@inline PureRNGs._cursor_ordinal(::TR.Tandem8x32, cursor::TR.Tandem8x32) = cursor.pos

# The Gamma fallback reads a child generator as its own scalar cursor.
@inline function PureRNGs._child_cursor(rng::TR.Tandem8x32, purpose::UInt64)
    child = TR.subrng(rng, purpose)
    return child, child
end

@noinline _span_exhausted() =
    throw(ArgumentError("draw span passes the end of the Tandem stream"))

# Check all factors before multiplying. Even an empty fill checks its alignment.
@inline function _checked_span(rng, draws::Integer, count::Integer, width::Val)
    slot = UInt64(_slot_bits(width))
    start = TR._align_up(rng.pos, Int(slot))
    start >= rng.pos || _span_exhausted()
    0 <= draws && 0 <= count <= typemax(UInt64) || _span_exhausted()
    available = (typemax(UInt64) - start) ÷ slot
    iszero(count) && return start, start
    draws <= available ÷ UInt64(count) || _span_exhausted()
    return start, start + UInt64(draws) * UInt64(count) * slot
end

@inline function PureRNGs._draw_cursor(rng::TR.Tandem8x32, count::Integer, width::Val)
    start, stop = _checked_span(rng, 1, count, width)
    return TR._advance(rng, start), TR._advance(rng, stop)
end

@inline function PureRNGs._reserve_draws(
    rng::TR.Tandem8x32,
    draws::Integer,
    count::Integer,
    width::Val,
)
    _, stop = _checked_span(rng, draws, count, width)
    return TR._advance(rng, stop)
end

@inline function PureRNGs._addressed_state(
    rng::TR.Tandem8x32,
    count::Integer,
    width::Val,
    i::Integer,
)
    i >= 1 || throw(ArgumentError("addressed draw index must be positive"))
    start, stop = _checked_span(rng, i, count, width)
    position = stop - UInt64(count) * UInt64(_slot_bits(width))
    return TR._advance(rng, position)
end

@inline function PureRNGs._fill_cursor(
    rng::TR.Tandem8x32,
    count::Integer,
    width::Val,
    ordinal::UInt64,
)
    start = TR._align_up(rng.pos, _slot_bits(width))
    position = start + ordinal * UInt64(count) * UInt64(_slot_bits(width))
    return TR._advance(rng, position)
end

# Scalar draws may wrap, as Tandem's native scalars do. The codec returns the
# successor cursor, including every reserved candidate, so no second skip is needed.
@inline function PureRNGs._engine_draw_next(rng::TR.Tandem8x32, codec, ::Type{T}) where {T}
    _, width = PureRNGs._codec_takes(codec, T)
    cursor = TR._advance(rng, TR._align_up(rng.pos, _slot_bits(width)))
    return PureRNGs._codec_take(codec, rng, cursor, T)
end

@inline function PureRNGs._engine_draw_at(
    rng::TR.Tandem8x32,
    codec,
    ::Type{T},
    i::Integer,
) where {T}
    count, width = PureRNGs._codec_takes(codec, T)
    addressed = PureRNGs._addressed_state(rng, count, width, i)
    return first(PureRNGs._codec_take(codec, addressed, addressed, T))
end

# Neighbouring Bools of a BitArray share a word, so its chunks cannot run on threads.
_bit_packed(destination) = destination isa BitArray
_bit_packed(destination::Union{SubArray,Base.ReshapedArray}) = _bit_packed(parent(destination))

@inline function _fill_codec_chunk!(
    rng,
    destination::AbstractArray{T},
    codec,
    first,
    last,
) where {T}
    count, width = PureRNGs._codec_takes(codec, T)
    cursor = _row_cursor(PureRNGs._fill_cursor(rng, count, width, UInt64(first - 1)))
    @inbounds for index = first:last
        value, cursor = PureRNGs._codec_take(codec, rng, cursor, T)
        destination[index] = value
    end
    return nothing
end

function _fill_codec!(
    rng::TR.Tandem8x32{K,TR._CPUBackend},
    destination,
    codec,
    threaded,
) where {K}
    count, width = PureRNGs._codec_takes(codec, eltype(destination))
    if count == 1 && codec isa Union{
        PureRNGs._NormalCodec,
        PureRNGs._ExponentialCodec,
        PureRNGs._MappedFillCodec,
        _ScaledCodec,
    }
        mapped = _MappedArray(destination, codec, width)
        nthreads = threaded && !_bit_packed(destination) ? Threads.nthreads() : 1
        TR.rand_fill!(rng, mapped; nthreads)
        return nothing
    end
    if threaded && !_bit_packed(destination)
        chunk = max(1, (8192 * 32) ÷ (count * _slot_bits(width)))
        PureRNGs._run_chunks(length(destination), chunk) do first, last
            _fill_codec_chunk!(rng, destination, codec, first, last)
        end
    else
        _fill_codec_chunk!(rng, destination, codec, 1, length(destination))
    end
    return nothing
end

include("purerngs_fill.jl")

function _fill_codec!(rng::TR.Tandem8x32, destination, codec, threaded)
    count, width = PureRNGs._codec_takes(codec, eltype(destination))
    TR._fill_mapped!(rng, destination, codec, count, width, Val(_slot_bits(width)))
    return nothing
end

function _fill_column_chunk!(rng, destination, codec, first, last)
    count, width = PureRNGs._codec_takes(codec, eltype(destination))
    cursor = _row_cursor(PureRNGs._fill_cursor(rng, count, width, UInt64(first - 1)))
    for column = first:last
        cursor = PureRNGs._column_take!(codec, rng, cursor, destination, column)
    end
    return nothing
end

function _fill_columns!(
    rng::TR.Tandem8x32{K,TR._CPUBackend},
    destination,
    codec,
    threaded,
) where {K}
    count, width = PureRNGs._codec_takes(codec, eltype(destination))
    if threaded
        chunk = max(1, (8192 * 32) ÷ (count * _slot_bits(width)))
        PureRNGs._run_chunks(size(destination, 2), chunk) do first, last
            _fill_column_chunk!(rng, destination, codec, first, last)
        end
    else
        _fill_column_chunk!(rng, destination, codec, 1, size(destination, 2))
    end
    return nothing
end

# A device column fill is the device element fill of the column codec.
_fill_columns!(rng::TR.Tandem8x32, destination, codec, threaded) =
    _fill_codec!(rng, destination, codec, threaded)

function PureRNGs._engine_fill_columns!(
    rng::TR.Tandem8x32,
    destination::AbstractMatrix{T},
    threaded::Bool,
    codec,
) where {T}
    Base.require_one_based_indexing(destination)
    count, width = PureRNGs._codec_takes(codec, T)
    next_rng = PureRNGs._reserve_draws(rng, size(destination, 2), count, width)
    isempty(destination) || _fill_columns!(rng, destination, codec, threaded)
    return destination, next_rng
end

function PureRNGs._engine_fill!(
    rng::TR.Tandem8x32,
    destination::AbstractArray{T},
    threaded::Bool,
    codec,
) where {T}
    Base.require_one_based_indexing(destination)
    count, width = PureRNGs._codec_takes(codec, T)
    next_rng = PureRNGs._reserve_draws(rng, length(destination), count, width)
    isempty(destination) || _fill_codec!(rng, destination, codec, threaded)
    return destination, next_rng
end

# A complex normal is two consecutive real takes, each scaled by sqrt(1/2). The
# real fill of the reinterpreted array with the scaled codec draws the same span
# with one take per element, which the row and tile paths serve directly.
struct _ScaledCodec{C}
    codec::C
end

@inline PureRNGs._fill_width(scaled::_ScaledCodec, ::Type{T}) where {T} =
    PureRNGs._fill_width(scaled.codec, T)
@inline PureRNGs._cooperative_value(scaled::_ScaledCodec, ::Type{T}, raw) where {T} =
    T(sqrt(0.5)) * PureRNGs._cooperative_value(scaled.codec, T, raw)

function PureRNGs._engine_fill!(
    rng::TR.Tandem8x32,
    destination::AbstractArray{Complex{T}},
    threaded::Bool,
    codec::PureRNGs._NormalCodec,
) where {T}
    components = reinterpret(reshape, T, destination)
    scaled = _ScaledCodec(codec)
    # Tasks split at chunk-group boundaries, which are multiples of two slots.
    # A pair that straddles one would reach its complex element from two tasks,
    # each rewriting the whole element, so such a start fills on one task.
    _, width = PureRNGs._codec_takes(scaled, T)
    slot = _slot_bits(width)
    threaded &= iszero(TR._align_up(rng.pos, slot) % (2slot))
    _, next_rng = PureRNGs._engine_fill!(rng, components, threaded, scaled)
    return destination, next_rng
end

for f in (:randn_next, :randn_next!, :randn_at, :randexp_next, :randexp_next!, :randexp_at)
    body = Symbol(:_engine_, f)
    @eval @inline PureRNGs.$f(rng::TR.Tandem8x32, args...; kwargs...) =
        PureRNGs.$body(rng, args...; kwargs...)
end

# Match the uniform forwarders' generator bound so their typed methods stay more
# specific. PureRNGs' generic distribution bodies do not serve traced Reactant
# values, so the backend lookup they start with says so.
PureRNGs._engine_backend(::TR._ReactantRNG) =
    throw(ArgumentError("PureRNGs sampler draws do not support Reactant states"))
@inline PureRNGs.rand_next(rng::Generator, d; kwargs...) =
    PureRNGs._engine_rand_next(rng, d; kwargs...)
@inline PureRNGs.rand_next(rng::Generator, d, dims::Integer...; kwargs...) =
    PureRNGs._engine_rand_next(rng, d, dims...; kwargs...)
@inline PureRNGs.rand_next(rng::Generator, d, dims::Dims; kwargs...) =
    PureRNGs._engine_rand_next(rng, d, dims; kwargs...)
@inline PureRNGs.rand_next!(rng::Generator, d, destination; kwargs...) =
    PureRNGs._engine_rand_next!(rng, d, destination; kwargs...)
@inline PureRNGs.rand_at(rng::Generator, d, i::Integer) =
    PureRNGs._engine_rand_at(rng, d, i)
