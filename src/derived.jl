# Bounded integers, normals, and exponentials derived from the uniform stream. Appendix A of
# the specification fixes them so that every port returns the same values: the bounded
# draws and their fallback keys are exact, and the normals and exponentials copy the C
# reference's polynomials, fused multiply-adds and ziggurat tables, so they are bit for bit
# equal to tandem-c.

# --- bounded integers ------------------------------------------------------------------------

# Reserved purposes of the fallback generators of bounded fills, by draw width.
const BELOW_PURPOSE_32 = UInt64(0x424c573332)
const BELOW_PURPOSE_64 = UInt64(0x424c573634)

@inline _below_purpose(::Type{UInt32}) = BELOW_PURPOSE_32
@inline _below_purpose(::Type{UInt64}) = BELOW_PURPOSE_64

# High and low w-bit halves of the 2w-bit product x·n.
@inline function _lemire(x::UInt32, n::UInt32)
    m = UInt64(x) * n
    return (m >> 32) % UInt32, m % UInt32
end
@inline function _lemire(x::UInt64, n::UInt64)
    m = UInt128(x) * n
    return (m >> 64) % UInt64, m % UInt64
end

# (2^w − n) mod n. Only reached when the low half is below n, so n is not zero.
@inline _threshold(n::U) where {U<:Unsigned} = (zero(U) - n) % n

@inline _draw(rng::Tandem8x32, ::Type{U}) where {U} = rand_next(rng, U)

# Lemire's method on the w-bit draws of `s`, rejecting by drawing the next w bits.
# n = 0 returns 0 after one draw.
@inline function _below(s, n::U) where {U<:Union{UInt32,UInt64}}
    x, s = _draw(s, U)
    hi, lo = _lemire(x, n)
    if lo < n
        t = _threshold(n)
        while lo < t
            x, s = _draw(s, U)
            hi, lo = _lemire(x, n)
        end
    end
    return hi, s
end

# An offset uniform on [0, span] with the width chosen from the range span + 1: 32 bits up
# to 2^32 values, else 64. A range of exactly 2^w values takes the w-bit draw itself, which
# is what Lemire's method gives for range 2^w.
@inline function _bounded_offset(s, span::UInt64)
    if span < typemax(UInt32)
        x, s = _below(s, (span + 1) % UInt32)
    elseif span == typemax(UInt32)
        x, s = _draw(s, UInt32)
    elseif span < typemax(UInt64)
        x, s = _below(s, span + 1)
    else
        x, s = _draw(s, UInt64)
    end
    return UInt64(x), s
end

# last − first of a non-empty range. The difference fits in the unsigned type of T.
@inline function _span(r::AbstractUnitRange{T}) where {T<:Base.BitInteger}
    isempty(r) && throw(ArgumentError("range must be non-empty"))
    d = (last(r) - first(r)) % unsigned(T)
    d <= typemax(UInt64) ||
        throw(ArgumentError("bounded draws support ranges of at most 2^64 values"))
    return d % UInt64
end

"""
    rand_next(rng, r::AbstractUnitRange{<:Integer}) -> (x, rng′)

Draw `x` uniform on the integers of `r` by Lemire's method (Appendix A of the
specification). The draw width follows the number of values: 32 bits up to 2^32 values,
else 64. The element type of `r` does not change the value, only its type. A rejected draw
retries on the next draw of the stream. Ranges of more than 2^64 values are not supported.
"""
@inline function rand_next(rng::Tandem8x32, r::AbstractUnitRange{T}) where {T<:Base.BitInteger}
    offset, rng = _bounded_offset(rng, _span(r))
    return first(r) + offset % T, rng
end

"""
    rand_below_next(rng, n::Union{UInt32,UInt64}) -> (x, rng′)

Draw `x` uniform on `0:n-1` by Lemire's method on draws of the width of `n`'s type, as
tandem-c's `tandem_u32_below` and `tandem_u64_below`. `n = 0` returns 0 after one draw.
"""
@inline rand_below_next(rng::Tandem8x32, n::Union{UInt32,UInt64}) = _below(rng, n)

# The retry of a rejected element of a bounded fill. The fallback generator depends on the
# key and the element's global draw index `g` only, so a fill cut at any element boundary
# equals the whole fill: split(g) of purpose(P_w) of the key at position 0.
@noinline function _below_retry(key::O4, ::Val{K}, n::U, g::UInt64) where {K,U}
    sub = _child_key(key, _below_purpose(U), DOMAIN_FOLD, UInt32(0), 0)
    child = Tandem8x32{K}(_child_key(sub, g >> 1, DOMAIN_SPLIT, UInt32(0), g & 1), 0)
    t = _threshold(n)
    while true
        x, child = rand_next(child, U)
        hi, lo = _lemire(x, n)
        lo >= t && return hi
    end
end

# Element i takes draw i of the plain fill, so the fill consumes exactly length(A) draws and
# rows fill independently. An empty fill leaves the position unchanged.
function _below_fill!(rng::Tandem8x32{K}, A::AbstractArray{U}, n::U) where {K,U}
    isempty(A) && return rng
    next_rng = rand_fill!(rng, A)
    first_draw = rngposition(next_rng) ÷ UInt64(8 * sizeof(U)) - UInt64(length(A))
    @inbounds for i = 1:length(A)
        hi, lo = _lemire(A[i], n)
        if lo < n && lo < _threshold(n)
            hi = _below_retry(rng.key, Val(K), n, first_draw + UInt64(i - 1))
        end
        A[i] = hi
    end
    return next_rng
end

"""
    rand_below_fill!(rng, A::AbstractArray{U}, n::U) -> rng′    (U = UInt32 or UInt64)

Fill `A` with integers uniform on `0:n-1`, as tandem-c's `tandem_fill_u32_below` and
`tandem_fill_u64_below`. Element `i` maps draw `i` of `rand_fill!(rng, A)` and the fill
consumes exactly `length(A)` draws. A rejected draw retries on the fallback generator
`splitrng` index `g` of `subrng(P_w)` of the key at position 0, where `g` is the draw's
index in the stream (Appendix A of the specification). A fill cut at any element boundary
therefore equals the whole fill. An empty fill leaves the position unchanged.
"""
function rand_below_fill!(rng::Tandem8x32, A::AbstractArray{U}, n::U) where {U<:Union{UInt32,UInt64}}
    _check_cpu_fill(rng, A)
    return _below_fill!(rng, A, n)
end

"""
    rand_fill!(rng, A::AbstractArray, r::AbstractUnitRange{<:Integer}) -> rng′

Fill `A` with integers uniform on `r`. The draw width follows the number of values as for
[`rand_next`](@ref), and the fill follows [`rand_below_fill!`](@ref): one draw per element,
rejected draws retry on their fallback generator, and an empty fill leaves the position
unchanged.
"""
function rand_fill!(
    rng::Tandem8x32,
    A::AbstractArray,
    r::AbstractUnitRange{T},
) where {T<:Base.BitInteger}
    _check_cpu_fill(rng, A)
    return _bounded_fill!(rng, A, first(r), _span(r))
end

# `Stateful` shares this path so both APIs return the same values and positions.
function _bounded_fill!(rng, A::AbstractArray, lo, span::UInt64)
    isempty(A) && return rng
    if span <= typemax(UInt32)
        return _range_fill!(rng, A, lo, (span + 1) % UInt32, span == typemax(UInt32))
    end
    return _range_fill!(rng, A, lo, span + 1, span == typemax(UInt64))
end

# `full` marks a range of exactly 2^w values, which takes the plain draws.
function _range_fill!(rng, A::AbstractArray, lo::T, n::U, full::Bool) where {T,U}
    raw = A isa Array{U} ? A : Vector{U}(undef, length(A))
    next_rng = full ? rand_fill!(rng, raw) : _below_fill!(rng, raw, n)
    if raw !== A || !iszero(lo)
        for (i, x) in zip(eachindex(A), raw)
            @inbounds A[i] = lo + x % T
        end
    end
    return next_rng
end

@inline function _check_cpu_fill(rng::Tandem8x32, A::AbstractArray)
    _check_fill_device(rng, A)
    rng.device isa _CPUBackend ||
        throw(ArgumentError("derived fills run on the CPU"))
    Base.require_one_based_indexing(A)
    return nothing
end

# --- normals and exponentials ----------------------------------------------------------------

# -2 ln x for x in (0, 1], the logarithm of the normals and the Float64 exponentials, as
# tandem-c.
# x = mant·2^k with mant in [sqrt(1/2), sqrt(2)): adding the bits of sqrt(1/2) to the
# exponent field makes the mantissa rollover pick k. Then ln mant = 2s(1 + z/3 + z²/5 + …)
# with s = (mant − 1)/(mant + 1) and z = s², and ln 2 is split so that nk·ln2_hi is exact.
# Julia does not contract a product into a sum, so the explicit fma calls fix every bit.
@inline function _neg2_log(x::Float64)
    ix = reinterpret(UInt64, x) + 0x00095f6200000000
    nk = Float64(1023 - (ix >> 52) % Int64)
    mant = reinterpret(Float64, (ix & 0x000fffffffffffff) + 0x3fe6a09e00000000)
    s = (mant - 1.0) / (mant + 1.0)
    z = s * s
    p = fma(z, 0.08312363319426472, 0.09070001083303751)
    p = fma(z, p, 0.11111433317907482)
    p = fma(z, p, 0.14285712049336274)
    p = fma(z, p, 0.2000000000566491)
    p = fma(z, p, 0.33333333333331017)
    p = fma(z, p, 1.0)
    return fma(nk, 3.816429394731813e-10, fma(nk, 1.3862943607382476, (s * -4.0) * p))
end

@inline function _neg2_log(x::Float32)
    ix = reinterpret(UInt32, x) + 0x004afb0d
    nk = Float32(Int32(127) - (ix >> 23) % Int32)
    mant = reinterpret(Float32, (ix & 0x007fffff) + 0x3f3504f3)
    s = (mant - 1.0f0) / (mant + 1.0f0)
    z = s * s
    p = fma(z, fma(z, fma(z, 0.14275366f0, 0.20000061f0), 0.33333334f0), 1.0f0)
    return fma(nk, 2.857213530660374f-6, fma(nk, 1.38629150390625f0, (s * -4.0f0) * p))
end

# Base.sqrt checks the sign and throws, which keeps a loop from vectorizing. The argument
# here is never negative.
@inline _sqrt(x::Union{Float32,Float64}) = Core.Intrinsics.sqrt_llvm(x)

# Rotate (cos, sin) of the reduced angle by q quarter turns: odd q swaps them, bit 1 of q
# negates the sine, and bit 1 of q + 1 negates the cosine.
@inline function _quarter_turn(cs::T, sn::T, q::S) where {T,S}
    U = Base.uinttype(T)
    qu = q % U
    shift = 8 * sizeof(T) - 2
    sign = one(U) << (8 * sizeof(T) - 1)
    swap = zero(U) - (qu & one(U))
    sb, cb = reinterpret(U, sn), reinterpret(U, cs)
    xb = ((sb & swap) | (cb & ~swap)) ⊻ (((qu + one(U)) << shift) & sign)
    yb = ((cb & swap) | (sb & ~swap)) ⊻ ((qu << shift) & sign)
    return reinterpret(T, xb), reinterpret(T, yb)
end

# Float32 Box-Muller from uniforms a and b: r = sqrt(-2 ln(1 - a)) and (r cos 2πb, r sin 2πb).
# The nearest quarter turn q leaves b − q/4 exact, so the angle in [−π/4, π/4] needs no further
# reduction, and short polynomials give cos and sin there.
@inline function _box_muller(a::Float32, b::Float32)
    r = _sqrt(_neg2_log(1.0f0 - a))
    q = unsafe_trunc(Int32, b * 4.0f0 + 0.5f0)
    f = fma(-Float32(q), 0.25f0, b)
    # 2π as a float pair, so the angle is good to the last bit of the float.
    th = fma(f, -1.7484555f-7, f * 6.2831855f0)
    w = th * th
    hs = fma(w, fma(w, fma(w, 2.72499f-6, -0.00019840087f0), 0.008333332f0), -0.16666667f0)
    hc = fma(w, fma(w, fma(w, 2.4463761f-5, -0.0013887589f0), 0.04166665f0), -0.5f0)
    c, s = _quarter_turn(fma(w, hc, 1.0f0), th * fma(w, hs, 1.0f0), q)
    return r * c, r * s
end

# −ln x for x in (0, 1], the Float32 exponentials, within 0.58 ulp for every 1 − x on the
# 2^−24 grid, as tandem-c's neg_log_f32. An error near 1 ulp moves 1 − exp(−ln x) to a
# neighbouring grid point, so the leading term u = (2 − 2m)/(m + 1) = −2s is carried as
# uh + r/d: m + 1 = d + dl exactly, and r is the residual of uh. nk·ln2_hi + uh splits exactly
# by fast two-sum, because nk·ln2_hi is exact and either 0 or larger than |uh|. uh rounds in an
# fma, so that no contraction feeds the unrounded num·rcp to the two-sum. The tail u³q(u²) is
# a minimax fit to 2 atanh(u/2) − u.
@inline function _neg_log(x::Float32)
    ix = reinterpret(UInt32, x) + 0x004afb0d
    nk = Float32(Int32(127) - (ix >> 23) % Int32)
    mant = reinterpret(Float32, (ix & 0x007fffff) + 0x3f3504f3)
    num = fma(mant, -2.0f0, 2.0f0)
    d = mant + 1.0f0
    dl = mant - (d - 1.0f0)
    rcp = 1.0f0 / d
    uh = fma(num, rcp, 0.0f0)
    r = fma(-uh, dl, fma(-uh, d, num))
    v = uh * uh
    q = fma(v, fma(v, 0.0023109776f0, 0.012496489f0), 0.08333336f0)
    a = nk * 0.693145751953125f0
    hi = a + uh
    e = uh - (hi - a)
    return hi + fma(uh * v, q, fma(r, rcp, fma(nk, 1.428606765330187f-6, e)))
end

# Halving −2 ln is exact.
@inline _exponential(u::Float64) = 0.5 * _neg2_log(1.0 - u)
@inline _exponential(u::Float32) = _neg_log(1.0f0 - u)

const NormalTypes = Union{Float32,Float64}

# Float64 normals are the 1024-layer ziggurat of Appendix A, one UInt64 draw per element. A
# draw that misses the inner rectangles continues on its own fallback generator, split(g) of
# purpose(NORMAL_PURPOSE_64) of the key at position 0, where g is the global index of the
# draw. So element i depends on draw i alone, and a fill cut anywhere equals the whole fill.
const NORMAL_PURPOSE_64 = UInt64(0x4e524d3634)

@inline _normal_sub(key::O4) = _child_key(key, NORMAL_PURPOSE_64, DOMAIN_FOLD, UInt32(0), 0)

# (±ra·W[i], ra < K[i]): the candidate of a draw and whether it lies in the inner rectangle.
# ZIG_W holds −W[i] at 1024 + i, so bit 10 picks the sign by the index. ra < 2^53 converts
# exactly.
@inline function _zig_candidate(r::UInt64)
    ra = r >> 11
    x = Float64(ra % Int64) * @inbounds(ZIG_W[(r&0x7ff)+1])
    return x, ra < @inbounds(ZIG_K[(r&0x3ff)+1])
end

# Draw d of a fallback generator from position 0. Draws 2j and 2j + 1 share block j of the
# stream, so each pair costs one `_block_at` instead of a row of eight.
@inline function _fallback_next(key::O4, blk::O4, d::Int, ::Val{K}) where {K}
    if iseven(d) && d >= 2
        blk = _block_at(key, UInt64(64d), K)
    end
    lo, hi = isodd(d) ? (blk[3], blk[4]) : (blk[1], blk[2])
    return UInt64(lo) | UInt64(hi) << 32, blk, d + 1
end

@inline _unit(d::UInt64) = Float64(d >> 11) * 0x1p-53

# The slow path of a draw r that missed, whose global draw index is g, on the fallback
# split(g) of `sub`.
@noinline function _zig_miss(sub::O4, ::Val{K}, r::UInt64, g::UInt64) where {K}
    key = _child_key(sub, g >> 1, DOMAIN_SPLIT, UInt32(0), g & 1)
    return _zig_slow(key, _block_at(key, UInt64(0), K), r, Val(K))
end

# The fallback keys and first blocks of the misses at global draw indices g, one per lane:
# the two F of `_zig_miss` eight wide.
@inline function _fallback_seed8(sub::O4, g::NTuple{8,UInt64})
    c = map(x -> x >> 1, g)
    lo, hi = Lane8(map(x -> x % UInt32, c)), Lane8(map(x -> (x >> 32) % UInt32, c))
    split = (lo, hi, Lane8(DOMAIN_SPLIT), Lane8(UInt32(0)))
    even, odd = seed(split, ntuple(w -> Lane8(sub[w]), Val(4)))
    key = ntuple(Val(4)) do w
        Lane8(ntuple(l -> isodd(g[l]) ? lane(odd[w], l) : lane(even[w], l), Val(8)))
    end
    zero8 = Lane8(UInt32(0))
    blk, _ = step(seed((zero8, zero8, Lane8(DOMAIN_STREAM), Lane8(AUX_STREAM)), key)...)
    return key, blk
end

@inline _lane4(t::NTuple{4,Lane8}, l::Int) = ntuple(w -> lane(t[w], l), Val(4))

# Resolve the first m ≤ 8 queued misses at indices ks of R, which shares A's storage and
# still holds their raw draws. Spare lanes repeat a queued index and go unused. On the M4
# the eight-wide seeding costs 0.85 of eight scalar ones, but eight misses back to back cost
# 0.6 of eight misses resolved one at a time.
@noinline function _zig_resolve!(
    A,
    R,
    ks::NTuple{8,Int},
    m::Int,
    sub::O4,
    g0::UInt64,
    ::Val{K},
) where {K}
    g = ntuple(l -> g0 + UInt64(ks[min(l, m)] - 1), Val(8))
    key, blk = _fallback_seed8(sub, g)
    for l = 1:m
        k = ks[l]
        @inbounds A[k] = _zig_slow(_lane4(key, l), _lane4(blk, l), R[k], Val(K))
    end
    return nothing
end

# ln is −0.5·_neg2_log, exact given _neg2_log. Julia does not contract a product into a sum,
# so every other operation rounds once, as the appendix requires.
@inline function _zig_slow(key::O4, blk::O4, r::UInt64, ::Val{K}) where {K}
    d = 0
    while true
        i = (r & 0x3ff) % Int
        x, inner = _zig_candidate(r)
        inner && return x
        if i == 0
            # The tail beyond R, by Marsaglia's method.
            while true
                u, blk, d = _fallback_next(key, blk, d, Val(K))
                a = 0.5 * _neg2_log(1.0 - _unit(u)) / ZIG_R
                u, blk, d = _fallback_next(key, blk, d, Val(K))
                b = 0.5 * _neg2_log(1.0 - _unit(u))
                b + b >= a * a && return isodd(r >> 10) ? -(ZIG_R + a) : ZIG_R + a
            end
        end
        u, blk, d = _fallback_next(key, blk, d, Val(K))
        y = @inbounds ZIG_Y[i+1] + _unit(u) * (ZIG_Y[i+2] - ZIG_Y[i+1])
        -0.5 * _neg2_log(y) < -0.5 * (x * x) && return x
        r, blk, d = _fallback_next(key, blk, d, Val(K))
    end
end

"""
    normal_next(rng, T = Float64) -> (z, rng′)

A standard normal of type `Float32` or `Float64` (Appendix A of the specification), element 1
of [`normal_fill!`](@ref). A Float64 normal is the 1024-layer ziggurat of one UInt64 draw. A
Float32 normal is the cosine half `sqrt(-2 log(1 - a)) cos(2πb)` of the Box-Muller pair of two
Float32 draws `a` and `b`, computed in Float32. The values equal tandem-c's
`tandem_normal_f64` and `tandem_normal_f32` bit for bit.
"""
normal_next(rng::Tandem8x32) = normal_next(rng, Float64)

@inline function normal_next(rng::Tandem8x32{K}, ::Type{Float64}) where {K}
    g = _align_up(rng.pos, 64) >> 6
    r, rng = rand_next(rng, UInt64)
    x, inner = _zig_candidate(r)
    return (inner ? x : _zig_miss(_normal_sub(rng.key), Val(K), r, g)), rng
end

@inline function normal_next(rng::Tandem8x32, ::Type{Float32})
    a, rng = rand_next(rng, Float32)
    b, rng = rand_next(rng, Float32)
    return _box_muller(a, b)[1], rng
end

"""
    exponential_next(rng, T = Float64) -> (e, rng′)

A standard exponential `-log(1 - u)` of type `Float32` or `Float64` from one uniform draw
`u` of type `T` (Appendix A of the specification), with the polynomial logarithm of the
normals. It equals element 1 of [`exponential_fill!`](@ref).
"""
@inline function exponential_next(rng::Tandem8x32, ::Type{T} = Float64) where {T<:NormalTypes}
    u, rng = rand_next(rng, T)
    return _exponential(u), rng
end

# Uniform-fill A[1:count] group by group and pass each run of completed elements to
# `map!(A, i, j)` while it is still in cache. Runs start after and end on multiples of
# `unit`. Groups are distributed over tasks as in `rand_fill!`. A unit that straddles two
# tasks is mapped after both have written it. With a `state`, each task threads its own copy
# through `state = map!(A, i, j, state)` and ends with `flush!(A, state)`.
function _fill_then_map!(map!::F, A, key, p0, count, unit, nthreads, k::Val) where {F}
    stateful(A, i, j, s) = (map!(A, i, j); s)
    flush!(A, s) = nothing
    return _fill_then_map!(stateful, A, key, p0, count, unit, nthreads, k, nothing, flush!)
end

function _fill_then_map!(
    map!::F,
    A::AbstractArray{T},
    key::O4,
    p0::UInt64,
    count::Int,
    unit::Int,
    nthreads::Integer,
    ::Val{K},
    state::S,
    flush!::G,
) where {F,T,K,S,G}
    nthreads >= 1 || throw(ArgumentError("nthreads must be at least 1"))
    nb = UInt64(draw_bits(T))
    pend = p0 + nb * UInt64(count)
    bits_per_group = UInt64(ROW_BITS) * UInt64(K)
    g0 = p0 ÷ bits_per_group
    g1 = (pend - 1) ÷ bits_per_group
    # Elements written once group g is complete.
    function written(g)
        start = g * bits_per_group
        return Int((start + min(bits_per_group, pend - start) - p0) ÷ nb)
    end
    function run(first_group, last_group)
        done = first_group == g0 ? 0 : cld(written(first_group - 1), unit) * unit
        s = state
        for g = first_group:last_group
            _fill_group!(A, key, g, p0, pend, Val(K))
            upto = written(g) ÷ unit * unit
            if upto > done
                s = map!(A, done + 1, upto, s)
            end
            done = max(done, upto)
        end
        flush!(A, s)
        return nothing
    end
    nt = min(Int(nthreads), Int(g1 - g0 + 1))
    if nt <= 1
        run(g0, g1)
        return nothing
    end
    ng = g1 - g0 + 1
    groups_per_task, extra = divrem(ng, UInt64(nt))
    first_group(t) = g0 + groups_per_task * UInt64(t) + min(UInt64(t), extra)
    Threads.@sync for t = 0:(nt-1)
        Threads.@spawn run(first_group(t), first_group(t + 1) - 1)
    end
    for t = 1:(nt-1)
        c = written(first_group(t) - 1)
        c % unit == 0 || flush!(A, map!(A, c - c % unit + 1, c - c % unit + unit, state))
    end
    return nothing
end

# LLVM loop metadata for the enclosing loop, placed as the last statement of its body.
macro _interleave(n::Int)
    return Expr(:loopinfo, (Symbol("llvm.loop.interleave.count"), n))
end

# Measured on the M4: four interleaved vector iterations hide the latency of the division
# and the square root. `@simd` alone was slower than no annotation.
@inline function _normal_pairs!(A::AbstractArray{T}, i::Int, j::Int) where {T}
    @inbounds for k = (i+1)÷2:j÷2
        z0, z1 = _box_muller(A[2k-1], A[2k])
        A[2k-1] = z0
        A[2k] = z1
        @_interleave 4
    end
    return nothing
end

@inline function _exponentials!(A::AbstractArray{T}, i::Int, j::Int) where {T}
    @inbounds for k = i:j
        A[k] = _exponential(A[k])
        @_interleave 4
    end
    return nothing
end

@inline function _check_span(p0::UInt64, pos::UInt64, nb::Int, draws::Int)
    p0 >= pos && UInt64(draws) <= (typemax(UInt64) - p0) ÷ UInt64(nb) ||
        throw(ArgumentError("fill of $draws draws passes the end of the stream"))
    return nothing
end

# Map the raw draws R[i:j] to normals in A[i:j]. R shares A's storage. The inner loop runs to
# the next miss and holds no call, which made the map 1.6 times faster on Zen 2 than a call
# in its body. It tests three draws at a time: on the M4 the in-place map ran 1.39 times
# faster than one at a time, and 1.07 and 1.09 times faster than two and four. A group of
# draws holds about two misses for K = 32, so the misses queue across groups in
# `q = (indices, count)` and resolve eight at a time, as in tandem-c.
@inline function _zig_map!(A, R, i::Int, j::Int, q, sub::O4, g0::UInt64, ::Val{K}) where {K}
    ks, m = q
    k = i
    @inbounds while k <= j
        while k + 2 <= j
            x1, inner1 = _zig_candidate(R[k])
            x2, inner2 = _zig_candidate(R[k+1])
            x3, inner3 = _zig_candidate(R[k+2])
            inner1 & inner2 & inner3 || break
            A[k] = x1
            A[k+1] = x2
            A[k+2] = x3
            k += 3
        end
        k > j && break
        x, inner = _zig_candidate(R[k])
        if inner
            A[k] = x
        else
            m += 1
            ks = Base.setindex(ks, k, m)
            if m == 8
                _zig_resolve!(A, R, ks, 8, sub, g0, Val(K))
                m = 0
            end
        end
        k += 1
    end
    return ks, m
end

"""
    normal_fill!(rng, A::AbstractArray{T}; nthreads = Threads.nthreads()) -> rng′

Fill `A` with standard normals (Appendix A of the specification). `T` is `Float32` or
`Float64`. The values equal tandem-c's `tandem_fill_normal_f64` and `tandem_fill_normal_f32`
bit for bit. Threads split the fill as in `rand_fill!` without changing the values.

- Float64: element `i` is the 1024-layer ziggurat of UInt64 draw `i`, and the fill consumes
  `n` draws. A draw that misses the inner rectangles continues on its own fallback generator,
  keyed by the draw's index in the stream, so a fill cut at any element equals the whole
  fill. An empty fill aligns the position to 64 bits.
- Float32: elements `2j − 1` and `2j` are the cosine and sine halves of the Box-Muller pair of
  uniform draws `2j − 1` and `2j` of `rand_fill!(rng, A)`. The fill consumes `2·cld(n, 2)`
  uniforms: an odd `n` writes only the cosine half of its last pair. An empty fill leaves
  the position unchanged.
"""
function normal_fill!(
    rng::Tandem8x32{K},
    A::AbstractArray{Float64};
    nthreads::Integer = Threads.nthreads(),
) where {K}
    _check_cpu_fill(rng, A)
    n = length(A)
    p0 = _align_up(rng.pos, 64)
    _check_span(p0, rng.pos, 64, n)
    n == 0 && return _advance(rng, p0)
    sub, g0 = _normal_sub(rng.key), p0 >> 6
    R = reinterpret(UInt64, A)
    map!(R, i, j, q) = _zig_map!(A, R, i, j, q, sub, g0, Val(K))
    flush!(R, (ks, m)) = m > 0 && _zig_resolve!(A, R, ks, m, sub, g0, Val(K))
    queue = (ntuple(_ -> 0, Val(8)), 0)
    _fill_then_map!(map!, R, rng.key, p0, n, 1, nthreads, Val(K), queue, flush!)
    return _advance(rng, p0 + UInt64(64) * UInt64(n))
end

function normal_fill!(
    rng::Tandem8x32{K},
    A::AbstractArray{T};
    nthreads::Integer = Threads.nthreads(),
) where {K,T<:Float32}
    _check_cpu_fill(rng, A)
    n = length(A)
    n == 0 && return rng
    nb = draw_bits(T)
    p0 = _align_up(rng.pos, nb)
    _check_span(p0, rng.pos, nb, n + isodd(n))
    even = n - isodd(n)
    even > 0 && _fill_then_map!(_normal_pairs!, A, rng.key, p0, even, 2, nthreads, Val(K))
    rng = _advance(rng, p0 + UInt64(nb) * UInt64(even))
    if isodd(n)
        @inbounds A[n], rng = normal_next(rng, T)
    end
    return rng
end

"""
    exponential_fill!(rng, A::AbstractArray{T}; nthreads = Threads.nthreads()) -> rng′

Fill `A` with standard exponentials: element `i` is `-log(1 - u)` for uniform draw `i` of
`rand_fill!(rng, A)` (Appendix A of the specification). `T` is `Float32` or `Float64`.
The fill consumes `length(A)` uniforms. An empty fill leaves the position unchanged.
Threads split the fill as in `rand_fill!` without changing the values.
"""
function exponential_fill!(
    rng::Tandem8x32{K},
    A::AbstractArray{T};
    nthreads::Integer = Threads.nthreads(),
) where {K,T<:NormalTypes}
    _check_cpu_fill(rng, A)
    n = length(A)
    n == 0 && return rng
    nb = draw_bits(T)
    p0 = _align_up(rng.pos, nb)
    _check_span(p0, rng.pos, nb, n)
    _fill_then_map!(_exponentials!, A, rng.key, p0, n, 1, nthreads, Val(K))
    return _advance(rng, p0 + UInt64(nb) * UInt64(n))
end
