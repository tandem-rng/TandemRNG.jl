# Array fills on the CPU. Real element i of a fill of type T sits
# at bit `align_up(pos, w) + w·(i − 1)` of the stream for the draw width w, so a fill and a
# loop of scalar draws agree by construction. Within each group of eight chunks, seed the
# eight lanes with F, run K eight-wide steps of T, and write the rows that fall inside the
# fill range. Complex fills use the corresponding real component array.

const V4 = NTuple{4,VecElement{UInt32}}

# The optional PureRNGs device extension owns mapped kernels. This hook keeps
# the two extensions independent of their load order.
function _fill_mapped!(rng, destination, codec, count, width, slot)
    throw(ArgumentError("load KernelAbstractions and GPUArraysCore to fill device arrays"))
end

@inline _v4(w::O4) =
    (VecElement(w[1]), VecElement(w[2]), VecElement(w[3]), VecElement(w[4]))

# One 16-byte block store. Base's `unsafe_store!` on a `Ptr` assumes no alignment, so the
# array may start at any byte of the stream. The GPU extension adds the device pointer method.
@inline _store_block!(p::Ptr, w::O4) = unsafe_store!(reinterpret(Ptr{V4}, p), _v4(w))

# Block `o` at element index `i` of an array that is not dense, one element at a time.
@inline function _store_block_at!(A::AbstractArray{T}, i::Int, o::O4) where {T}
    nb = draw_bits(T)
    for q = 0:(BLOCK_BITS÷nb-1)
        @inbounds A[i+q] = _element(T, o, UInt64(nb * q))
    end
    return nothing
end

"""
    rand_fill!(rng, A::AbstractArray{T}; nthreads = Threads.nthreads()) -> rng′

Fill `A` with values of type `T` and return the advanced generator. `T` is one of the draw
types of [`rand_next`](@ref). The destination backend must match the generator's binding,
including for empty arrays. Groups of eight chunks are distributed over `nthreads` tasks
when there are enough of them, and the result does not depend on the thread count.

On a GPU array (with KernelAbstractions and GPUArraysCore loaded) the fill is queued on the
array's device like other array operations and the call returns before it completes.
Synchronize the device before reading the array from the host.
"""
function rand_fill!(
    rng::Tandem8x32{K},
    A::AbstractArray{T};
    nthreads::Integer = Threads.nthreads(),
) where {K,T<:ScalarTypes}
    nthreads >= 1 || throw(ArgumentError("nthreads must be at least 1"))
    _check_fill_device(rng, A)
    _check_serviceability(rng, T)
    rng.device isa _CPUBackend ||
        throw(ArgumentError("load the GPU backend to fill device arrays"))
    Base.require_one_based_indexing(A)
    nb = draw_bits(T)
    n = UInt64(length(A))
    p0 = _align_up(rng.pos, nb)
    p0 >= rng.pos && n <= (typemax(UInt64) - p0) ÷ UInt64(nb) ||
        throw(ArgumentError("fill of $(length(A)) elements passes the end of the stream"))
    n == 0 && return _advance(rng, p0)
    pend = p0 + UInt64(nb) * n
    bits_per_group = UInt64(ROW_BITS) * UInt64(K)
    g0 = p0 ÷ bits_per_group
    g1 = (pend - 1) ÷ bits_per_group
    nt = min(Int(nthreads), Int(g1 - g0 + 1))
    if nt <= 1
        for g = g0:g1
            _fill_group!(A, rng.key, g, p0, pend, Val(K))
        end
    else
        # Each task writes adjacent groups, which keeps its stores local on NUMA hosts.
        ng = g1 - g0 + 1
        groups_per_task, extra = divrem(ng, UInt64(nt))
        Threads.@sync for t = 0:(nt-1)
            first_group = g0 + groups_per_task * UInt64(t) + min(UInt64(t), extra)
            last_group = first_group + groups_per_task + (t < extra) - 1
            Threads.@spawn for g = first_group:last_group
                _fill_group!(A, rng.key, g, p0, pend, Val(K))
            end
        end
    end
    return _advance(rng, pend)
end

function rand_fill!(
    rng::Tandem8x32,
    A::AbstractArray{Complex{T}};
    nthreads::Integer = Threads.nthreads(),
) where {T<:FloatTypes}
    _check_fill_device(rng, A)
    _check_serviceability(rng, Complex{T})
    return rand_fill!(rng, reinterpret(reshape, T, A); nthreads)
end

function rand_next(rng::Tandem8x32, ::Type{T}, dims::Dims) where {T<:DrawTypes}
    destination = _allocate_draw_array(rng, T, dims)
    return destination, rand_fill!(rng, destination)
end

const _UniformRNG = Union{Tandem8x32,_ReactantRNG}
rand_next(rng::_UniformRNG) = rand_next(rng, Float64)
rand_next(rng::_UniformRNG, dims::Dims) = rand_next(rng, Float64, dims)
rand_next(rng::_UniformRNG, ::Type{T}, n::Integer, dims::Integer...) where {T<:DrawTypes} =
    rand_next(rng, T, Int.((n, dims...)))
rand_next(rng::_UniformRNG, n::Integer, dims::Integer...) =
    rand_next(rng, Float64, Int.((n, dims...)))

function _fill_group!(
    A::AbstractArray{T},
    key::O4,
    g::UInt64,
    p0::UInt64,
    pend::UInt64,
    ::Val{K},
) where {T,K}
    prow = g * (UInt64(ROW_BITS) * UInt64(K))
    # Test once per complete group. Float32 keeps the bounded loop: the separate loop
    # loses throughput on both measured CPU architectures.
    if T !== Float32 && p0 <= prow && UInt64(ROW_BITS) * UInt64(K) <= pend - prow
        return _fill_full_group!(A, key, g, p0, Val(K))
    end
    o, h = seed_row(key, g)
    for _ = 1:K
        prow >= pend && break
        o, h = step(o, h)
        if prow >= p0 && UInt64(ROW_BITS) <= pend - prow
            _write_row!(A, o, prow, p0)
        elseif prow >= p0 || p0 - prow < UInt64(ROW_BITS)
            _write_partial_row!(A, o, prow, p0, pend)
        end
        prow += ROW_BITS
    end
    return nothing
end

# Keep the full loop separate so its registers do not share the partial-row branches.
@noinline function _fill_full_group!(A, key, g, p0, ::Val{K}) where {K}
    o, h = seed_row(key, g)
    prow = g * (UInt64(ROW_BITS) * UInt64(K))
    for _ = 1:K
        o, h = step(o, h)
        _write_row!(A, o, prow, p0)
        prow += ROW_BITS
    end
    return nothing
end

# The 128 bytes of a row in memory order as four vectors: the element transform applied to
# whole word vectors, then the 4×8 register transpose from word-major to lane-major.
@inline _row_words(::Type{T}, o::Row) where {T<:Integer} =
    _vtranspose(o[1].v, o[2].v, o[3].v, o[4].v)
@inline _row_words(::Type{Float32}, o::Row) =
    _vtranspose(_vfloat32(o[1].v), _vfloat32(o[2].v), _vfloat32(o[3].v), _vfloat32(o[4].v))

# Every nonzero Float16 result is normal and exact. Convert the integer to Float32,
# keep its ten fraction bits, and subtract 112 for the exponent bias plus 11 for scaling.
# Keeping the conversion in UInt32 vectors avoids scalar Float16 narrowing on AVX2.
@inline function _half_bits(raw::UInt16)
    n = UInt32(raw >> 5)
    bits = reinterpret(UInt32, Float32(n)) >> 13
    return ifelse(iszero(n), UInt16(0), (bits - UInt32(123 << 10)) % UInt16)
end

@inline _half_row_word(raw::UInt32) =
    UInt32(_half_bits(raw % UInt16)) | (UInt32(_half_bits((raw >> 16) % UInt16)) << 16)

@inline function _row_words(::Type{Float16}, o::Row)
    transform(v) = ntuple(i -> VecElement(_half_row_word(v[i].value)), Val(8))
    return _vtranspose(
        transform(o[1].v),
        transform(o[2].v),
        transform(o[3].v),
        transform(o[4].v),
    )
end
# (raw >> 11) · 2^-53 for the four little-endian word pairs of a memory-order vector.
@static if Sys.ARCH === :aarch64
    # ucvtf with 53 fraction bits converts and scales in one instruction. The value is
    # below 2^53, so the result is exact like the product.
    @inline _vfloat64(a::V8) = Base.llvmcall(
        (
            """
            declare <2 x double> @llvm.aarch64.neon.vcvtfxu2fp.v2f64.v2i64(<2 x i64>, i32)
            define <8 x i32> @entry(<8 x i32> %0) #0 {
                %w = bitcast <8 x i32> %0 to <4 x i64>
                %s = lshr <4 x i64> %w, <i64 11, i64 11, i64 11, i64 11>
                %a = shufflevector <4 x i64> %s, <4 x i64> poison, <2 x i32> <i32 0, i32 1>
                %b = shufflevector <4 x i64> %s, <4 x i64> poison, <2 x i32> <i32 2, i32 3>
                %fa = call <2 x double> @llvm.aarch64.neon.vcvtfxu2fp.v2f64.v2i64(<2 x i64> %a, i32 53)
                %fb = call <2 x double> @llvm.aarch64.neon.vcvtfxu2fp.v2f64.v2i64(<2 x i64> %b, i32 53)
                %f = shufflevector <2 x double> %fa, <2 x double> %fb, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
                %r = bitcast <4 x double> %f to <8 x i32>
                ret <8 x i32> %r
            }
            attributes #0 = { alwaysinline }""",
            "entry",
        ),
        V8,
        Tuple{V8},
        a,
    )
else
    @inline function _vfloat64(a::V8)
        raws = reinterpret(NTuple{4,VecElement{UInt64}}, a)
        values = ntuple(Val(4)) do i
            raw = raws[i].value
            @static if Sys.ARCH === :x86_64
                # AVX2 lacks vector UInt64 conversion. Construct the high 52 bits, then
                # restore bit 53 exactly. This is the same mapping as (raw >> 11) * 2^-53.
                top52 = reinterpret(Float64, 0x3ff0000000000000 | (raw >> 12)) - 1.0
                x = top52 + ifelse(iszero(raw & 0x800), 0.0, 0x1p-53)
            else
                x = Float64(raw >> 11) * 0x1p-53
            end
            VecElement(x)
        end
        return reinterpret(V8, values)
    end
end

@inline function _row_words(::Type{Float64}, o::Row)
    words = _vtranspose(o[1].v, o[2].v, o[3].v, o[4].v)
    return map(_vfloat64, words)
end

# A row wholly inside the range. A dense array takes it as four 32-byte stores (unaligned:
# Base's `unsafe_store!` on a `Ptr` assumes no alignment), other arrays block by block.
@inline _write_row!(A::Array, o::Row, prow::UInt64, p0::UInt64) =
    _write_dense_row!(A, o, prow, p0)

# Reinterpreting a dense complex array preserves contiguous component storage.
@inline _write_row!(
    A::Base.ReinterpretArray{T,N,Complex{T},<:Array},
    o::Row,
    prow::UInt64,
    p0::UInt64,
) where {T,N} = _write_dense_row!(A, o, prow, p0)

# The Float64 normal fill writes its raw draws into the destination's own storage.
@inline _write_row!(
    A::Base.ReinterpretArray{UInt64,N,Float64,<:Array},
    o::Row,
    prow::UInt64,
    p0::UInt64,
) where {N} = _write_dense_row!(A, o, prow, p0)

@inline function _write_dense_row!(
    A::AbstractArray{T},
    o::Row,
    prow::UInt64,
    p0::UInt64,
) where {T}
    T === Char && return _write_row_elements!(A, o, prow, p0)
    first = Int((prow - p0) ÷ UInt64(draw_bits(T))) + 1
    p = reinterpret(Ptr{V8}, pointer(A, first))
    v = _row_words(T, o)
    unsafe_store!(p, v[1], 1)
    unsafe_store!(p, v[2], 2)
    unsafe_store!(p, v[3], 3)
    unsafe_store!(p, v[4], 4)
    return nothing
end

# A Bool row is 1024 elements, one byte each: every word of every lane expands to 32 bytes.
@inline function _write_row!(A::Array{Bool}, o::Row, prow::UInt64, p0::UInt64)
    first = Int(prow - p0) + 1
    p = reinterpret(Ptr{V32}, pointer(A, first))
    for ℓ = 0:(LANES-1), w = 1:4
        unsafe_store!(p, _vexpand(lane(o[w], ℓ + 1)), 4ℓ + w)
    end
    return nothing
end

@inline function _write_row!(
    A::AbstractArray{T},
    o::Row,
    prow::UInt64,
    p0::UInt64,
) where {T}
    return _write_row_elements!(A, o, prow, p0)
end

@inline function _write_row_elements!(
    A::AbstractArray{T},
    o::Row,
    prow::UInt64,
    p0::UInt64,
) where {T}
    nb = draw_bits(T)
    per_block = BLOCK_BITS ÷ nb
    first = Int((prow - p0) ÷ UInt64(nb)) + 1
    for ℓ = 0:(LANES-1)
        _store_block_at!(A, first + ℓ * per_block, _block(o, ℓ))
    end
    return nothing
end

@inline function _write_partial_row!(
    A::AbstractArray{T},
    o::Row,
    prow::UInt64,
    p0::UInt64,
    pend::UInt64,
) where {T}
    nb = draw_bits(T)
    for ℓ = 0:(LANES-1)
        block = _block(o, ℓ)
        for q = 0:(BLOCK_BITS÷nb-1)
            b = prow + UInt64(BLOCK_BITS * ℓ + nb * q)
            if p0 <= b < pend
                A[Int((b-p0)÷UInt64(nb))+1] = _element(T, block, b)
            end
        end
    end
    return nothing
end
