# Weighted choice of Appendix C of the specification: Walker's alias method with a table
# built in exact integers, so every port returns the same table and the same indices.

"""
    ChoiceTable(weights::AbstractVector{<:Real})

Alias table for drawing an index `i` in `1:m` with probability proportional to
`weights[i]` (Appendix C of the specification). Each weight is converted to `Float64`. The
weights must be finite and not negative, at least one must be positive, and
`1 ≤ m < 2^32`. Other input throws an `ArgumentError`. The build uses exact integer
arithmetic and consumes no draws, so the table equals tandem-c's `tandem_choice_build`.

Fields: `capacity` is the column capacity `S`, `cut[j]` the part of column `j` kept by `j`,
and `alias[j]` the one-based index that takes the rest.
"""
struct ChoiceTable
    capacity::UInt64
    cut::Vector{UInt64}
    alias::Vector{UInt32}

    function ChoiceTable(weights::AbstractVector{<:Real})
        w = Float64[x for x in weights]
        m = length(w)
        1 <= m <= typemax(UInt32) ||
            throw(ArgumentError("a choice needs between 1 and 2^32 - 1 weights"))
        all(x -> isfinite(x) && x >= 0, w) ||
            throw(ArgumentError("choice weights must be finite and not negative"))
        wmax = maximum(w)
        wmax > 0 || throw(ArgumentError("a choice needs a positive weight"))
        return new(_choice_table!(w, wmax)...)
    end
end

_nbits(x::UInt64) = 64 - leading_zeros(x)

# ceil(w·2^t) for finite w ≥ 0, exact, from the integer significand of w: w = s·2^e. ldexp
# alone would round a positive w to 0 where the product is subnormal. A shift of 64 or more
# leaves q = 0 and r = s, so a tiny positive w gives 1. The caller keeps the result below 2^64.
function _ceil_scaled(w::Float64, t::Int)
    iszero(w) && return UInt64(0)
    b = reinterpret(UInt64, w)
    f = b & 0x000fffffffffffff
    biased = (b >> 52) % Int
    s, e = biased == 0 ? (f, -1074) : (f | 0x0010000000000000, biased - 1075)
    k = e + t
    k >= 0 && return s << k
    q, r = s >> -k, s & ((UInt64(1) << -k) - 1)
    return q + !iszero(r)
end

# The first pass at a scale that cannot overflow bounds the total, and the second puts it
# just below 2^63. Then Vose's pairing in place: cut holds the masses until a column is
# paired. Index l is one-based here, so the C loop bounds shift by one.
function _choice_table!(w::Vector{Float64}, wmax::Float64)
    m = length(w)
    t = 63 - _nbits(UInt64(m)) - exponent(wmax)
    total = sum(x -> _ceil_scaled(x, t), w; init = UInt64(0))
    t += 63 - _nbits(total)
    cut = [_ceil_scaled(x, t) for x in w]
    total = sum(cut)
    big = argmax(cut)
    s = cld(total, UInt64(m))
    cut[big] += s * UInt64(m) - total
    alias = collect(UInt32(1):UInt32(m))
    l = findfirst(>=(s), cut)
    for i = 1:m
        j = i
        while j <= i && cut[j] < s
            alias[j] = l
            cut[l] -= s - cut[j]
            j = l
            if cut[l] < s
                l += 1
                while l <= m && cut[l] < s
                    l += 1
                end
            end
        end
    end
    return s, cut, alias
end

@inline function _choice(t::ChoiceTable, r::UInt64)
    x = UInt128(r) * UInt64(length(t.cut))
    j = (x >> 64) % Int + 1
    v = ((x % UInt64) * UInt128(t.capacity)) >> 64
    return @inbounds v < t.cut[j] ? j : Int(t.alias[j])
end

"""
    choice_next(rng, t::ChoiceTable) -> (i, rng′)

Draw a one-based index `i` from the alias table `t` (Appendix C of the specification). The
draw consumes one UInt64 draw and equals element 1 of [`choice_fill!`](@ref) and tandem-c's
`tandem_choice` plus 1.
"""
@inline function choice_next(rng::Tandem8x32, t::ChoiceTable)
    r, rng = rand_next(rng, UInt64)
    return _choice(t, r), rng
end

"""
    choice_fill!(rng, A::AbstractArray{<:Integer}, t::ChoiceTable; nthreads = Threads.nthreads()) -> rng′

Fill `A` with one-based indices drawn from the alias table `t`, as tandem-c's
`tandem_fill_choice` plus 1. Element `i` comes from UInt64 draw `i` of the plain fill, so the
fill consumes `length(A)` draws and a fill cut at any element equals the whole fill. An
empty fill aligns the position to 64 bits. The element type of `A` must hold the number of
weights. Threads split the fill as in `rand_fill!` without changing the values.
"""
function choice_fill!(
    rng::Tandem8x32{K},
    A::AbstractArray{T},
    t::ChoiceTable;
    nthreads::Integer = Threads.nthreads(),
) where {K,T<:Base.BitInteger}
    _check_cpu_fill(rng, A)
    length(t.cut) <= typemax(T) ||
        throw(ArgumentError("element type $T cannot hold $(length(t.cut)) indices"))
    n = length(A)
    p0 = _align_up(rng.pos, 64)
    _check_span(p0, rng.pos, 64, n)
    n == 0 && return _advance(rng, p0)
    # A 64-bit array takes the raw draws in place.
    R = A isa Array && sizeof(T) == 8 ? reinterpret(UInt64, A) : Vector{UInt64}(undef, n)
    function map!(R, i, j)
        @inbounds for k = i:j
            A[k] = _choice(t, R[k]) % T
        end
    end
    _fill_then_map!(map!, R, rng.key, p0, n, 1, nthreads, Val(K))
    return _advance(rng, p0 + UInt64(64) * UInt64(n))
end
