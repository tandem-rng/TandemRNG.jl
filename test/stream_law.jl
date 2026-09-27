# Check stream order, alignment, reconstruction, and fill invariance across thread counts.

const DRAW_TYPES =
    (Bool, UInt8, Int8, UInt16, Int16, UInt32, Int32, Float32, UInt64, Int64, Float64)

# Bits one draw consumes: a Bool takes one bit, everything else its size.
bits(::Type{T}) where {T} = T === Bool ? 1 : 8 * sizeof(T)

function scalar_loop(rng, ::Type{T}, n) where {T}
    out = Vector{T}(undef, n)
    for i = 1:n
        out[i], rng = rand_next(rng, T)
    end
    return out, rng
end

@testset "stream law: fill equals scalar loop" begin
    # Starts: fresh, one bit in (after a Bool), one byte in (after a UInt8). Both offsets
    # leave every wider fill misaligned to the 128-bit blocks.
    for K in (1, 32), T in DRAW_TYPES, start in (nothing, Bool, UInt8)
        rng = Tandem8x32{K}(7)
        start === nothing || (rng = rand_next(rng, start)[2])
        n = 3 * 1024 * K ÷ bits(T) + 5   # three groups and a partial block
        loop, rng_loop = scalar_loop(rng, T, n)
        filled = Vector{T}(undef, n)
        rng_fill = rand_fill!(rng, filled)
        @test filled == loop
        @test rngposition(rng_fill) == rngposition(rng_loop)
    end
end

@testset "stream law: short-fill continuation" begin
    key = rngkey(Tandem8x32(42))
    # Same row, next row, next chunk group, and the reconstruction path beyond one row.
    for (position, count) in ((265, 80), (1928, 100), (32 * 1024 - 512, 128), (1024, 129))
        rng = Tandem8x32(key, position)
        expected, scalar = scalar_loop(rng, UInt8, count)
        values = similar(expected)
        after = rand_fill!(rng, values; nthreads = 1)
        @test values == expected
        @test rngposition(after) == rngposition(scalar)
        @test rand_next(after, Float64) == rand_next(scalar, Float64)
    end
end

@testset "stream law: natural alignment" begin
    rng = Tandem8x32(11)
    _, after_bool = rand_next(rng, Bool)
    x, _ = rand_next(after_bool, Float64)
    @test x == rand_at(rng, Float64, 2)
    @test rngposition(after_bool) == 1
    _, after_float32 = rand_next(rng, Float32)
    y, after_both = rand_next(after_float32, Float64)
    @test y == rand_at(rng, Float64, 2)
    @test rngposition(after_both) == 128
    # Eight Bools in a row consume one byte's worth of bits, each a different bit.
    bools = scalar_loop(rng, Bool, 8)[1]
    w, _ = rand_next(rng, UInt8)
    @test bools == [isodd(w >> i) for i = 0:7]
end

@testset "stream law: random access" begin
    rng = Tandem8x32{8}(3)
    n = 3 * 8 * 16 + 1
    A = Vector{Float64}(undef, n)
    rand_fill!(rng, A)
    @test all(rand_at(rng, Float64, i) == A[i] for i = 1:n)
    B = Vector{UInt32}(undef, 2n)
    rand_fill!(rng, B)
    @test all(rand_at(rng, UInt32, i) == B[i] for i = 1:2n)
    C = Vector{Bool}(undef, 64n)
    rand_fill!(rng, C)
    @test all(rand_at(rng, Bool, i) == C[i] for i = 1:64n)
end

@testset "stream law: mixed scalar boundaries" begin
    for K in (1, 32, 64), pos in (1, 63, 127, 1023, 1024 * K - 1)
        rng = Tandem8x32{K}(rngkey(Tandem8x32(42)), pos)
        for T in DRAW_TYPES
            expected = rand_at(rng, T, 1)
            value, rng = @inferred rand_next(rng, T)
            @test value == expected
            @test rng == Tandem8x32{K}(rngkey(rng), rngposition(rng))
        end
    end
end

@testset "stream law: position round-trip" begin
    rng = Tandem8x32{4}(5)
    n = 1200
    A = Vector{UInt8}(undef, n)
    rand_fill!(rng, A)
    # Byte offsets: mid-block, block boundary, row boundary (128 bytes), group boundary (4
    # rows), and a later group mid-block. Positions count bits.
    for byte in (3, 16, 128, 512, 1030)
        rebuilt = Tandem8x32{4}(rngkey(rng), 8 * byte)
        rest, _ = scalar_loop(rebuilt, UInt8, n - byte)
        @test rest == A[(byte+1):end]
    end
end

@testset "stream law: fixture" begin
    # Test vectors from the independent UInt64 scalar oracle.
    # A change here changes the stream identity.
    rng = Tandem8x32(42)
    f, _ = rand_next(rng, Float64)
    u, _ = rand_next(rng, UInt32)
    b, _ = rand_next(rng, Bool)
    @test rngkey(rng) == (0x421d21eb, 0x32d31777, 0x62e7564b, 0xdf2bdf82)
    @test f == 0.9829130398628935
    @test u == 0x05e80cec
    @test b == false
    # Block 1 (lane 1, a different chunk) and block 8 (row 1) pin the row layout.
    @test rand_at(rng, Float64, 3) == 0.47759300283385586
    @test rand_at(rng, Float64, 17) == 0.9692135305890753
    @test rand_at(rng, UInt32, 5) == 0xf847db66
    # Bools are bits of the same words: element 32 is bit 31 of the first word.
    @test rand_at(rng, Bool, 32) == isodd(0x05e80cec >> 31)
    @test rand_at(rng, Bool, 129) == isodd(0xf847db66)
end

@testset "stream law: final partial row" begin
    # An advanced stream may pass the constructor's 2^63 start limit. Keep a padded
    # parent to detect a full-row store beyond the one-word destination safely.
    rng = TandemRNG._advance(Tandem8x32(42), typemax(UInt64) - 1023)
    expected, next = rand_next(rng, UInt32)
    storage = fill(UInt32(0xbadc0ffe), 33)
    after = rand_fill!(rng, view(storage, 1:1); nthreads = 1)
    @test storage[1] == expected
    @test all(==(0xbadc0ffe), view(storage, 2:33))
    @test rngposition(after) == rngposition(next)
    tail = TandemRNG._advance(rng, typemax(UInt64))
    @test_throws ArgumentError rand_fill!(tail, Vector{UInt32}(undef, 1))
    @test_throws ArgumentError rand_fill!(tail, Vector{Bool}(undef, 1))
end

@testset "stream law: block words equal elements" begin
    # The 16-byte image of a block that the block stores write must be the element sequence.
    o = (0x9e3779b9, 0x00000000, 0xffffffff, 0x7fc0a501)
    for T in DRAW_TYPES
        T === Bool && continue
        s = sizeof(T)
        elements = [TandemRNG._element(T, o, UInt64(8 * s * q)) for q = 0:(16÷s-1)]
        words = collect(TandemRNG._block_words(T, o))
        @test reinterpret(UInt32, elements) == words
    end
end

@testset "fill: thread invariance" begin
    rng = Tandem8x32{16}(9)
    _, rng = rand_next(rng, UInt32)   # start mid-block
    n = 3 * 512 + 7                   # three groups of Float32 and a partial block
    A = Vector{Float32}(undef, n)
    B = Vector{Float32}(undef, n)
    ra = rand_fill!(rng, A; nthreads = 1)
    # Four groups split across three tasks also exercise unequal contiguous ranges.
    rb = rand_fill!(rng, B; nthreads = 3)
    @test A == B
    @test rngposition(ra) == rngposition(rb)
end

@testset "bounds: start position" begin
    key = rngkey(Tandem8x32(1))
    @test Tandem8x32(key, UInt64(1) << 63 - 128) isa Tandem8x32
    @test_throws ArgumentError Tandem8x32(key, UInt64(1) << 63)
end
