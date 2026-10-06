module TandemRNGReactantExt

import TandemRNG as TR
import Reactant
using PrecompileTools: @setup_workload, @compile_workload

const Ops = Reactant.Ops
const Number = Reactant.TracedRNumber
const Array = Reactant.TracedRArray
const RNG = TR._ReactantRNG
const DrawTypes =
    Union{Bool,UInt8,Int8,UInt16,Int16,UInt32,Int32,UInt64,Int64,Float16,Float32,Float64}

@inline _constant(x::Number{T}, value) where {T} = Ops.constant(T(value))
@inline _constant(x::Array{T}, value) where {T} =
    Reactant.broadcast_to_size(Ops.constant(T(value)), size(x))
@inline _convert(::Type{T}, x::Number) where {T} = Ops.convert(Number{T}, x)
@inline _convert(::Type{T}, x::Array{S,N}) where {T,S,N} = Ops.convert(Array{T,N}, x)
@inline _vector(x::Number) = Reactant.broadcast_to_size(x, (1,))
@inline _shl(x, n::Integer) = Ops.shift_left(x, _constant(x, n))
@inline _shr(x, n::Integer) = Ops.shift_right_logical(x, _constant(x, n))
@inline _and(x, n::Integer) = Ops.and(x, _constant(x, n))
@inline _add(x, n::Integer) = Ops.add(x, _constant(x, n))

# Scalar and vector words share Tandem's single recurrence implementation.
struct Word{A}
    value::A
end
@inline Base.:⊻(a::Word, b::Word) = Word(Ops.xor(a.value, b.value))
@inline Base.:⊻(a::Word, b::UInt32) = Word(Ops.xor(a.value, _constant(a.value, b)))
@inline Base.:|(a::Word, b::UInt32) = Word(Ops.or(a.value, _constant(a.value, b)))
@inline Base.:+(a::Word, b::UInt32) = Word(_add(a.value, b))
@inline TR.rotl(a::Word, ::Val{R}) where {R} =
    Word(Ops.or(_shl(a.value, R), _shr(a.value, 32 - R)))
@inline function TR.mulwide(a::Word, b::Word)
    product = Ops.multiply(_convert(UInt64, a.value), _convert(UInt64, b.value))
    return Word(_convert(UInt32, _shr(product, 32))), Word(_convert(UInt32, product))
end

function Reactant.to_rarray(rng::TR.Tandem8x32{K}; kwargs...) where {K}
    state = Reactant.to_rarray(UInt64[TR.rngkey(rng)..., TR.rngposition(rng)]; kwargs...)
    return RNG{K,typeof(state)}(state)
end

function TR.Tandem8x32(rng::RNG{K}) where {K}
    state = Base.Array(rng.state)
    key = ntuple(i -> state[i] % UInt32, Val(4))
    pos = state[5]
    o, h = TR._state_at(TR._key64(key)..., pos, Val(K))
    return TR.Tandem8x32{K}(key, pos, o, h)
end

@inline _state_value(rng::RNG, i) = Reactant.@allowscalar rng.state[i]
TR.rngkey(rng::RNG) = ntuple(i -> _convert(UInt32, _state_value(rng, i)), Val(4))
TR.rngposition(rng::RNG) = _state_value(rng, 5)
TR.chunk_length(::RNG{K}) where {K} = K

@inline function _state(::Val{K}, key, pos) where {K}
    parts = [_vector(_convert(UInt64, x)) for x in key]
    push!(parts, _vector(pos))
    state = Ops.concatenate(parts, 1)
    return RNG{K,typeof(state)}(state)
end
@inline _advance(rng::RNG{K}, pos) where {K} = _state(Val(K), TR.rngkey(rng), pos)
@inline _align(pos, width) = _and(_add(pos, width - 1), ~(UInt64(width) - 1))

@inline function _seed(key, counter, domain::UInt32, aux::UInt32)
    o = (
        Word(_convert(UInt32, counter)),
        Word(_convert(UInt32, _shr(counter, 32))),
        Word(_constant(key[1], domain)),
        Word(_constant(key[1], aux)),
    )
    return TR.seed(o, map(Word, key))
end

function _block(rng::RNG{K}, pos) where {K}
    block = _shr(pos, 7)
    row = _shr(block, 3)
    counter = Ops.or(_shl(_shr(row, trailing_zeros(K)), 3), _and(block, 7))
    o, h = _seed(TR.rngkey(rng), counter, TR.DOMAIN_STREAM, TR.AUX_STREAM)
    limit = _and(row, K - 1)
    Reactant.@trace for i = UInt64(0):limit
        o, h = TR.step(o, h)
    end
    return map(x -> x.value, o)
end

@inline function _select(words, index)
    result = words[end]
    for i = (length(words)-1):-1:1
        condition = Ops.compare(index, _constant(index, i-1); comparison_direction = "EQ")
        result = Ops.select(condition, words[i], result)
    end
    return result
end

@inline _value(::Type{Float32}, raw) = Ops.multiply(
    _convert(Float32, _shr(raw, 8)),
    _constant(_convert(Float32, raw), 0x1p-24),
)
@inline _value(::Type{Float16}, raw) = _convert(
    Float16,
    Ops.multiply(
        _convert(Float32, _shr(_convert(UInt16, raw), 5)),
        _constant(_convert(Float32, raw), Float32(0x1p-11)),
    ),
)
@inline _value(::Type{Float64}, raw) = Ops.multiply(
    _convert(Float64, _shr(raw, 11)),
    _constant(_convert(Float64, raw), 0x1p-53),
)
@inline _value(::Type{Bool}, raw) =
    Ops.compare(_and(raw, 1), _constant(raw, 1); comparison_direction = "EQ")
@inline _value(::Type{T}, raw::Number) where {T<:Signed} =
    Ops.bitcast_convert(T, _convert(unsigned(T), raw))
@inline _value(::Type{T}, raw::Array{S,N}) where {T<:Signed,S,N} =
    Ops.bitcast_convert(Array{T,N}, _convert(unsigned(T), raw))
@inline _value(::Type{T}, raw) where {T<:Unsigned} = _convert(T, raw)

@inline function _element(::Type{T}, words, pos) where {T}
    index = _and(_shr(pos, 5), 3)
    lo = _select(words, index)
    raw = if TR.draw_bits(T) == 64
        hi = _select(words, _add(index, 1))
        Ops.or(_convert(UInt64, lo), _shl(_convert(UInt64, hi), 32))
    else
        Ops.shift_right_logical(lo, _convert(UInt32, _and(pos, 31)))
    end
    return _value(T, raw)
end

function TR.rand_next(rng::RNG, ::Type{T}) where {T<:DrawTypes}
    width = TR.draw_bits(T)
    pos = _align(TR.rngposition(rng), width)
    return _element(T, _block(rng, pos), pos), _advance(rng, _add(pos, width))
end

function TR.rand_next(rng::RNG, ::Type{T}, dims::Dims) where {T<:DrawTypes}
    destination = similar(rng.state, T, dims)
    return destination, TR.rand_fill!(rng, destination)
end

# `eltype` of a traced destination names its traced scalar type.
TR.rand_next(rng::RNG, ::Type{Number{T}}) where {T<:DrawTypes} = TR.rand_next(rng, T)

function TR.rand_at(rng::RNG, ::Type{T}, i::Integer) where {T<:DrawTypes}
    i >= 1 || throw(ArgumentError("index must be at least 1"))
    width = TR.draw_bits(T)
    pos = _add(_align(TR.rngposition(rng), width), UInt64(width) * (UInt64(i) - 1))
    return _element(T, _block(rng, pos), pos)
end
TR.rand_at(rng::RNG, ::Type{Number{T}}, i::Integer) where {T<:DrawTypes} =
    TR.rand_at(rng, T, i)

# One seed per chunk. The loop writes complete rows in the CPU stream order.
@inline function _write_row(buffer, o, j)
    groups = size(buffer, 4)
    row = Ops.concatenate([Ops.reshape(x.value, 1, 8, 1, groups) for x in o], 1)
    return Ops.dynamic_update_slice(buffer, row, [1, 1, j, 1])
end

function _stream_words(rng::RNG{K}, pos, ::Val{G}) where {K,G}
    chunks = 8G
    group = _shr(pos, 10 + trailing_zeros(K))
    counter = Ops.add(
        Reactant.broadcast_to_size(_shl(group, 3), (chunks,)),
        Ops.iota(UInt64, [chunks]; iota_dimension = 1),
    )
    key = map(x -> Reactant.broadcast_to_size(x, (chunks,)), TR.rngkey(rng))
    o, h = _seed(key, counter, TR.DOMAIN_STREAM, TR.AUX_STREAM)
    buffer = Reactant.broadcast_to_size(Ops.constant(UInt32(0)), (4, 8, K, G))
    Reactant.@trace for j = 1:K
        o, h = TR.step(o, h)
        buffer = _write_row(buffer, o, j)
    end
    return Ops.reshape(buffer, 32K * G), _shl(group, 10 + trailing_zeros(K))
end

function TR.rand_fill!(
    rng::RNG{K},
    destination::Array{T};
    nthreads::Integer = Threads.nthreads(),
) where {K,T<:DrawTypes}
    nthreads >= 1 || throw(ArgumentError("nthreads must be at least 1"))
    width = TR.draw_bits(T)
    pos = _align(TR.rngposition(rng), width)
    n = length(destination)
    n == 0 && return _advance(rng, pos)
    groups = cld(n * width + 1024K - 1, 1024K)
    words, start = _stream_words(rng, pos, Val(groups))
    offsets = Ops.add(
        Reactant.broadcast_to_size(Ops.subtract(pos, start), (n,)),
        Ops.multiply(
            Ops.iota(UInt64, [n]; iota_dimension = 1),
            Reactant.broadcast_to_size(Ops.constant(UInt64(width)), (n,)),
        ),
    )
    indices = _add(_shr(offsets, 5), 1)
    lo = words[indices]
    raw = if width == 64
        hi = words[_add(indices, 1)]
        Ops.or(_convert(UInt64, lo), _shl(_convert(UInt64, hi), 32))
    else
        Ops.shift_right_logical(lo, _convert(UInt32, _and(offsets, 31)))
    end
    values = Ops.reshape(_value(T, raw), size(destination)...)
    Reactant.TracedUtils.set_mlir_data!(destination, values.mlir_data)
    return _advance(rng, _add(pos, UInt64(n) * UInt64(width)))
end

function _child(rng::RNG{K}, counter, domain, aux::UInt32, half::Integer) where {K}
    o, h = _seed(TR.rngkey(rng), counter, domain, aux)
    key = map(x -> x.value, iszero(half) ? o : h)
    return _state(Val(K), key, Ops.constant(UInt64(0)))
end
TR.splitrng(rng::RNG) = TR.splitrng(rng, Val(2))
function TR.splitrng(rng::RNG, ::Val{N}) where {N}
    (N isa Int && N >= 0) || throw(ArgumentError("N must be a non-negative Int"))
    return ntuple(Val(N)) do i
        _child(rng, Ops.constant(UInt64((i-1) >> 1)), TR.DOMAIN_SPLIT, UInt32(0), (i-1) & 1)
    end
end
TR.subrng(rng::RNG, purpose::Integer) =
    _child(rng, Ops.constant(purpose % UInt64), TR.DOMAIN_FOLD, UInt32(0), 0)
function TR.forkrng(rng::RNG, ::Val{N}) where {N}
    (N isa Int && 0 <= N <= TR.MAX_FORK_CHILDREN) ||
        throw(ArgumentError("invalid child count"))
    block = _shr(TR.rngposition(rng), 7)
    children = ntuple(Val(N)) do i
        _child(rng, block, TR.DOMAIN_FORK, UInt32((i-1) >> 1), (i-1) & 1)
    end
    return _advance(rng, _shl(_add(block, 1), 7)), children
end
function TR.forkrng(rng::RNG)
    next_rng, children = TR.forkrng(rng, Val(1))
    return next_rng, only(children)
end

@setup_workload begin
    # Cache Julia host dispatch without creating a device client or XLA executable.
    @compile_workload for K in (1, 32, 64)
        precompile(Reactant.to_rarray, (TR.Tandem8x32{K,TR._CPUBackend},))
        traced = RNG{K,Array{UInt64,1}}
        for T in (
            Bool,
            UInt8,
            Int8,
            UInt16,
            Int16,
            UInt32,
            Int32,
            UInt64,
            Int64,
            Float16,
            Float32,
            Float64,
        )
            precompile(TR.rand_next, (traced, Type{T}))
            precompile(TR.rand_at, (traced, Type{T}, Int))
            for N in (1, 2)
                precompile(TR.rand_next, (traced, Type{T}, NTuple{N,Int}))
                precompile(TR.rand_fill!, (traced, Array{T,N}))
            end
        end
        precompile(TR.splitrng, (traced,))
        precompile(TR.splitrng, (traced, Val{2}))
        precompile(TR.subrng, (traced, Int))
        precompile(TR.forkrng, (traced,))
        precompile(TR.forkrng, (traced, Val{2}))
    end
end

end
