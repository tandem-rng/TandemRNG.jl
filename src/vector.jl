# Eight-lane vector primitives. `V8` is an LLVM `<8 x i32>`.
# LLVM operations express packed arithmetic, products, rotations, shuffles, and Bool expansion.
# Float conversion uses native Julia.

const V8 = NTuple{8,VecElement{UInt32}}

for (name, op) in ((:_vxor, "xor"), (:_vadd, "add"), (:_vor, "or"), (:_vand, "and"))
    ir = "%r = $op <8 x i32> %0, %1\nret <8 x i32> %r"
    @eval @inline $name(a::V8, b::V8) = Base.llvmcall($ir, V8, Tuple{V8,V8}, a, b)
end

@inline _vsplat(x::UInt32) = ntuple(_ -> VecElement(x), Val(8))

const V16 = NTuple{16,VecElement{UInt32}}

# High and low words of the eight 32x32 to 64-bit products.
@static if Sys.ARCH === :x86_64
    # AVX2 takes the high words from a widening product and the low words from vpmulld.
    # Unzipping both from one eight-lane 64-bit product costs lane-crossing extends and
    # permutes, which halved the fills on Zen 2.
    @inline function _vmulwide(a::V8, b::V8)
        hi = Base.llvmcall(
            """
            %a = zext <8 x i32> %0 to <8 x i64>
            %b = zext <8 x i32> %1 to <8 x i64>
            %p = mul <8 x i64> %a, %b
            %s = lshr <8 x i64> %p, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
            %r = trunc <8 x i64> %s to <8 x i32>
            ret <8 x i32> %r""",
            V8,
            Tuple{V8,V8},
            a,
            b,
        )
        lo = Base.llvmcall("%r = mul <8 x i32> %0, %1\nret <8 x i32> %r", V8, Tuple{V8,V8}, a, b)
        return hi, lo
    end
else
    # Both halves unzip from one widening product. A separate low multiply costs a vector
    # multiply per word on NEON.
    @inline _vmulwide(a::V8, b::V8) = _vmulwide_unzip(a, b)
end

@inline function _vmulwide_unzip(a::V8, b::V8)
    p = Base.llvmcall(
        """
        %a = zext <8 x i32> %0 to <8 x i64>
        %b = zext <8 x i32> %1 to <8 x i64>
        %p = mul <8 x i64> %a, %b
        %w = bitcast <8 x i64> %p to <16 x i32>
        ret <16 x i32> %w""",
        V16,
        Tuple{V8,V8},
        a,
        b,
    )
    return _vhalf(p, Val((1, 3, 5, 7, 9, 11, 13, 15))), _vhalf(p, Val((0, 2, 4, 6, 8, 10, 12, 14)))
end

@generated function _vhalf(a::V16, ::Val{M}) where {M}
    mask = join(("i32 $m" for m in M), ", ")
    ir = "%r = shufflevector <16 x i32> %0, <16 x i32> poison, <8 x i32> <$mask>\nret <8 x i32> %r"
    return :(Base.llvmcall($ir, V8, Tuple{V16}, a))
end

@generated function _vrotl(a::V8, ::Val{R}) where {R}
    if R == 16
        # A half-word swap is one instruction (NEON rev32), where shifts take two.
        ir = """
            %h = bitcast <8 x i32> %0 to <16 x i16>
            %s = shufflevector <16 x i16> %h, <16 x i16> poison, <16 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
            %r = bitcast <16 x i16> %s to <8 x i32>
            ret <8 x i32> %r"""
        return :(Base.llvmcall($ir, V8, Tuple{V8}, a))
    end
    left = join(("i32 $R" for _ = 1:8), ", ")
    right = join(("i32 $(32 - R)" for _ = 1:8), ", ")
    ir = """
        %s = shl <8 x i32> %0, <$left>
        %t = lshr <8 x i32> %0, <$right>
        %r = or <8 x i32> %s, %t
        ret <8 x i32> %r"""
    return :(Base.llvmcall($ir, V8, Tuple{V8}, a))
end

# Lane permutations. `M` is the LLVM mask: indices 0 to 7 pick from `a`, 8 to 15 from `b`.
@generated function _vshuffle(a::V8, b::V8, ::Val{M}) where {M}
    mask = join(("i32 $m" for m in M), ", ")
    ir = "%r = shufflevector <8 x i32> %0, <8 x i32> %1, <8 x i32> <$mask>\nret <8 x i32> %r"
    return :(Base.llvmcall($ir, V8, Tuple{V8,V8}, a, b))
end

@inline _vrotate1(a::V8) = _vshuffle(a, a, Val((1, 2, 3, 4, 5, 6, 7, 0)))

# Word-major row (word w of lanes 0 to 7) to memory order (lanes 0 to 7, each its four
# words). Output vector k holds blocks 2k and 2k + 1.
@static if Sys.ARCH === :x86_64
    # Eight two-source shuffles. AVX2 lowers the regrouping of the version below to eight
    # vpermq and four blends, which cost Zen 2 about 7 % of the fill.
    @inline function _vtranspose(w0::V8, w1::V8, w2::V8, w3::V8)
        lo = Val((0, 8, 1, 9, 2, 10, 3, 11))
        hi = Val((4, 12, 5, 13, 6, 14, 7, 15))
        p0 = _vshuffle(w0, w1, lo)
        p1 = _vshuffle(w0, w1, hi)
        q0 = _vshuffle(w2, w3, lo)
        q1 = _vshuffle(w2, w3, hi)
        pair_lo = Val((0, 1, 8, 9, 2, 3, 10, 11))
        pair_hi = Val((4, 5, 12, 13, 6, 7, 14, 15))
        return (
            _vshuffle(p0, q0, pair_lo),
            _vshuffle(p0, q0, pair_hi),
            _vshuffle(p1, q1, pair_lo),
            _vshuffle(p1, q1, pair_hi),
        )
    end
else
    # A 4×4 transpose inside each 128-bit half, then a regrouping of the halves. Every
    # shuffle but the last stays inside the halves, one NEON instruction each.
    @inline _vtranspose(w0::V8, w1::V8, w2::V8, w3::V8) = _vtranspose_halves(w0, w1, w2, w3)
end

@inline function _vtranspose_halves(w0::V8, w1::V8, w2::V8, w3::V8)
    zip_lo = Val((0, 8, 1, 9, 4, 12, 5, 13))
    zip_hi = Val((2, 10, 3, 11, 6, 14, 7, 15))
    t0 = _vshuffle(w0, w1, zip_lo)
    t1 = _vshuffle(w2, w3, zip_lo)
    t2 = _vshuffle(w0, w1, zip_hi)
    t3 = _vshuffle(w2, w3, zip_hi)
    pair_lo = Val((0, 1, 8, 9, 4, 5, 12, 13))
    pair_hi = Val((2, 3, 10, 11, 6, 7, 14, 15))
    b0 = _vshuffle(t0, t1, pair_lo)   # blocks 0 and 4
    b1 = _vshuffle(t0, t1, pair_hi)   # blocks 1 and 5
    b2 = _vshuffle(t2, t3, pair_lo)   # blocks 2 and 6
    b3 = _vshuffle(t2, t3, pair_hi)   # blocks 3 and 7
    low = Val((0, 1, 2, 3, 8, 9, 10, 11))
    high = Val((4, 5, 6, 7, 12, 13, 14, 15))
    return (
        _vshuffle(b0, b1, low),
        _vshuffle(b2, b3, low),
        _vshuffle(b0, b1, high),
        _vshuffle(b2, b3, high),
    )
end

# The 32 bits of one word as 32 bytes of 0 or 1, bit i in byte i. A Bool consumes one bit.
# Each source byte is repeated eight times, masked with the eight bit weights, and compared.
const V32 = NTuple{32,VecElement{UInt8}}

@inline _vexpand(x::UInt32) = Base.llvmcall(
    """
    %b = bitcast i32 %0 to <4 x i8>
    %e = shufflevector <4 x i8> %b, <4 x i8> poison, <32 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
    %m = and <32 x i8> %e, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
    %c = icmp ne <32 x i8> %m, zeroinitializer
    %r = zext <32 x i1> %c to <32 x i8>
    ret <32 x i8> %r""",
    V32,
    Tuple{UInt32},
    x,
)

# (raw >> 8) · 2^-24 per lane, as float bits.
@static if Sys.ARCH === :aarch64
    # ucvtf with 24 fraction bits converts and scales in one instruction. The value is
    # below 2^24, so the result is exact like the product.
    @inline _vfloat32(a::V8) = Base.llvmcall(
        (
            """
            declare <4 x float> @llvm.aarch64.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32>, i32)
            define <8 x i32> @entry(<8 x i32> %0) #0 {
                %s = lshr <8 x i32> %0, <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
                %a = shufflevector <8 x i32> %s, <8 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
                %b = shufflevector <8 x i32> %s, <8 x i32> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
                %fa = call <4 x float> @llvm.aarch64.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32> %a, i32 24)
                %fb = call <4 x float> @llvm.aarch64.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32> %b, i32 24)
                %f = shufflevector <4 x float> %fa, <4 x float> %fb, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
                %r = bitcast <8 x float> %f to <8 x i32>
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
    # Keep float lanes together until the final bitcast so Julia can vectorize the conversion.
    # The value is below 2^24, so the signed conversion is exact. AVX2 has one instruction for
    # it and none for the unsigned one.
    @inline _vfloat32(a::V8) = reinterpret(
        V8,
        ntuple(i -> VecElement(Float32((a[i].value >> 8) % Int32) * Float32(0x1p-24)), Val(8)),
    )
end
