# Bounded integers, normals, and exponentials of Appendix A of the specification against
# the fixtures of tandem-c and tandem-cuda, and the decomposition and consumption rules
# they share.

include("fixtures.jl")

# The C fixtures start from seed 42 after one Bool draw, at bit position 1.
after_bool(seed) = rand_next(Tandem8x32(seed), Bool)[2]

function scalar_draws(f, rng, count)
    x, rng = f(rng)
    out = [x]
    for _ = 2:count
        x, rng = f(rng)
        push!(out, x)
    end
    return out, rng
end

@testset "fixture files" begin
    # Byte identical to tandem-c b049384 (cross_below.h, cross_normal.h,
    # cross_exponential.h) and tandem-cuda c5c5725 (cross_fill_below.h,
    # cross_fill_exponential.h).
    @test fixture_sha256("cross_below.h") ==
          "0119fa58cc98d2da41d140408aaab57264166c73dc5ab22c1b0b725e282ae8f1"
    @test fixture_sha256("cross_fill_below.h") ==
          "d6d7a9dfcb42d02746e58ff99325c07fd4280f88fa59fcd780a200f65f8814bc"
    @test fixture_sha256("cross_normal.h") ==
          "e313b2f1cda2301f8c67cfae952219d4898df9a0623372965c39f6bb0edc7003"
    @test fixture_sha256("cross_exponential.h") ==
          "da848bae24dae7d1cde6fdb7ef2e6d2953b76b333ba5139800cba3ae03c85efc"
    @test fixture_sha256("cross_fill_exponential.h") ==
          "2a65543dbf94486201ca9310d1ffe328393ca60ab2c53aea8022d5ba739de7b0"
end

@testset "bounded: scalar draws equal tandem-c" begin
    text = fixture_text("cross_below.h")
    for (name, U) in (("CROSS_U32", UInt32), ("CROSS_U64", UInt64)), row in c_initializer(text, name)
        n = parse(U, row[1])
        want = parse.(U, row[2])
        got, rng = scalar_draws(r -> rand_below_next(r, n), after_bool(42), length(want))
        @test got == want
        @test rngposition(rng) == parse(UInt64, row[3])
        # The range interface picks the width from the range, so it reproduces the u32 rows
        # and the u64 rows with more than 2^32 values, whatever the result type.
        if U === UInt32 || n > UInt64(1) << 32
            r = Int128(-7):(Int128(-7)+n-1)
            got, _ = scalar_draws(rng -> rand_next(rng, r), after_bool(42), 64)
            @test got == Int128(-7) .+ want
        end
    end
end

@testset "bounded: fills equal tandem-cuda" begin
    text = fixture_text("cross_fill_below.h")
    key = Tuple(parse.(UInt32, c_initializer(text, "CROSS_FILL_KEY")))
    rejected = 0
    for (name, U, at) in (
        ("CROSS_BELOW32", UInt32, false),
        ("CROSS_BELOW64", UInt64, false),
        ("CROSS_BELOW32_AT", UInt32, true),
        ("CROSS_BELOW64_AT", UInt64, true),
    )
        for row in c_initializer(text, name)
            start, row = at ? (parse(Int, row[1]), row[2:end]) : (0, row)
            n = parse(U, row[1])
            want = parse.(U, row[3])
            rejected += parse(Int, row[2])
            A = Vector{U}(undef, length(want))
            rng = rand_below_fill!(Tandem8x32(key, start), A, n)
            @test A == want
            @test rngposition(rng) == rngposition(rand_fill!(Tandem8x32(key, start), A))
            if U === UInt32 || n > UInt64(1) << 32
                B = Vector{Int128}(undef, length(want))
                rand_fill!(Tandem8x32(key, start), B, Int128(-7):(Int128(-7)+n-1))
                @test B == Int128(-7) .+ want
            end
        end
    end
    @test rejected > 0
end

@testset "bounded: a cut fill equals the whole fill" begin
    # Ranges just above half the draw width reject about half the draws.
    key = rngkey(Tandem8x32(3))
    for (U, n) in ((UInt32, UInt32(2^31 + 1)), (UInt64, UInt64(2)^63 + 0x1)), start in (1, 12345)
        w = 8 * sizeof(U)
        whole = Vector{U}(undef, 300)
        rand_below_fill!(Tandem8x32(key, start), whole, n)
        p0 = cld(start, w) * w
        for cut in (1, 7, 100, 299)
            head = Vector{U}(undef, cut)
            tail = Vector{U}(undef, 300 - cut)
            rand_below_fill!(Tandem8x32(key, start), head, n)
            rand_below_fill!(Tandem8x32(key, p0 + w * cut), tail, n)
            @test vcat(head, tail) == whole
        end
    end
end

@testset "bounded: width from the range and degenerate ranges" begin
    rng = after_bool(11)
    # 2^32 values draw 32 bits, 2^32 + 1 values draw 64.
    x, next = rand_next(rng, 0:(Int64(2)^32-1))
    @test (x, next) == rand_next(rng, UInt32)
    _, next = rand_next(rng, 0:Int64(2)^32)
    @test rngposition(next) == 128
    @test rand_next(rng, typemin(Int64):typemax(Int64))[1] ==
          rand_next(rng, UInt64)[1] % Int64 + typemin(Int64)
    # Range 0 of the width-naming interface returns 0 after one draw.
    @test rand_below_next(rng, UInt32(0)) == (0x00000000, rand_next(rng, UInt32)[2])
    @test rand_below_next(rng, UInt64(0)) == (UInt64(0), rand_next(rng, UInt64)[2])
    @test_throws ArgumentError rand_next(rng, 1:0)
end

@testset "empty derived fills leave the position unchanged" begin
    rng = after_bool(5)
    @test rand_below_fill!(rng, UInt32[], UInt32(10)) == rng
    @test rand_fill!(rng, Int[], 1:10) == rng
    @test normal_fill!(rng, Float64[]) == rng
    @test exponential_fill!(rng, Float32[]) == rng
    # A full-width range takes the plain fill, which would align the position.
    st = Stateful(rng)
    rand!(st, Int32[], typemin(Int32):typemax(Int32))
    @test Tandem8x32(st) == rng
end

@testset "normals: pairs equal tandem-c bit for bit" begin
    text = fixture_text("cross_normal.h")
    for (T, name) in ((Float64, "CROSS_NORMAL"), (Float32, "CROSS_NORMALF"))
        want = parse.(T, c_initializer(text, name))
        got, rng = scalar_draws(after_bool(42), length(want) ÷ 2) do r
            pair = Vector{T}(undef, 2)
            return pair, normal_fill!(r, pair)
        end
        @test reduce(vcat, got) == want
        @test rngposition(rng) == parse(UInt64, c_scalar(text, name * "_END_POS"))
    end
end

@testset "normals: 1e6-pair dump equals tandem-c bit for bit" begin
    # tandem-c tools/dump_normals.c: seed (2026, 7), 2e6 − 1 f64 then 2e6 − 1 f32 normals from
    # one generator at each start.
    ctx = SHA.SHA256_CTX()
    d = Vector{Float64}(undef, 1_999_999)
    f = Vector{Float32}(undef, 1_999_999)
    for start in (0, 1, 77, 12345, 1 << 30)
        rng = Tandem8x32(rngkey(Tandem8x32(2026 + UInt128(7) << 64)), start)
        rng = normal_fill!(rng, d)
        SHA.update!(ctx, reinterpret(UInt8, d))
        normal_fill!(rng, f)
        SHA.update!(ctx, reinterpret(UInt8, f))
    end
    @test bytes2hex(SHA.digest!(ctx)) ==
          "cfae418807a7d5f91ecd3e42c33a00943690c6e4b888ee39206738783efe9ded"
end

@testset "exponentials equal tandem-c bit for bit" begin
    text = fixture_text("cross_exponential.h")
    key = rngkey(Tandem8x32(42))
    for (T, name) in ((Float64, "CROSS_EXPONENTIAL"), (Float32, "CROSS_EXPONENTIALF"))
        for row in c_initializer(text, name)
            rng = Tandem8x32(key, parse(Int, row[1]))
            want = parse.(T, row[2])
            A = Vector{T}(undef, length(want))
            @test rngposition(exponential_fill!(rng, A)) == parse(UInt64, row[3])
            @test A == want
        end
    end
    # tandem-cuda's host and device fills, the same key, at unaligned starts.
    text = fixture_text("cross_fill_exponential.h")
    for (T, name) in ((Float64, "CROSS_EXP64"), (Float32, "CROSS_EXP32"))
        for row in c_initializer(text, name)
            want = parse.(T, row[3])
            @test length(want) == parse(Int, row[2])
            A = Vector{T}(undef, length(want))
            exponential_fill!(Tandem8x32(key, parse(Int, row[1])), A)
            @test A == want
        end
    end
    # tandem-c tools/dump_exponentials.c: seed (2026, 7), 1e6 f64 then 1e6 f32 exponentials
    # from one generator at each start.
    ctx = SHA.SHA256_CTX()
    d = Vector{Float64}(undef, 1_000_000)
    f = Vector{Float32}(undef, 1_000_000)
    for start in (0, 1, 77, 12345, 1 << 30)
        rng = Tandem8x32(rngkey(Tandem8x32(2026 + UInt128(7) << 64)), start)
        rng = exponential_fill!(rng, d)
        SHA.update!(ctx, reinterpret(UInt8, d))
        exponential_fill!(rng, f)
        SHA.update!(ctx, reinterpret(UInt8, f))
    end
    @test bytes2hex(SHA.digest!(ctx)) ==
          "5c035a4ef1368231d25a9c2f9201be2df3224e28a14549a50625d0db3770ef4e"
end

@testset "normals and exponentials: draw consumption and decomposition" begin
    for T in (Float32, Float64), K in (1, 32)
        w = 8 * sizeof(T)
        rng = Tandem8x32{K}(rngkey(Tandem8x32(8)), 1)
        # An odd fill writes the cosine half of its last pair and consumes both uniforms.
        odd = Vector{T}(undef, 2001)
        even = Vector{T}(undef, 2002)
        @test rngposition(normal_fill!(rng, odd)) == rngposition(normal_fill!(rng, even))
        @test odd == even[1:end-1]
        @test normal_next(rng, T) == (even[1], rand_next(rand_next(rng, T)[2], T)[2])
        # A range that starts at an even element is the matching part of the whole fill.
        part = Vector{T}(undef, 1000)
        normal_fill!(Tandem8x32{K}(rngkey(rng), w + w * 1000), part)
        @test part == even[1001:2000]
        # One uniform per exponential.
        e = Vector{T}(undef, 2001)
        @test rngposition(exponential_fill!(rng, e)) == rngposition(rand_fill!(rng, similar(e)))
        @test exponential_next(rng, T)[1] == e[1]
        # An exponential fill cut at any element equals the whole fill.
        for cut in (1, 7, 1000)
            head = Vector{T}(undef, cut)
            tail = Vector{T}(undef, 2001 - cut)
            exponential_fill!(rng, head)
            exponential_fill!(Tandem8x32{K}(rngkey(rng), w + w * cut), tail)
            @test vcat(head, tail) == e
        end
        # Threads split the fill without changing the values.
        @test normal_fill!(rng, similar(even); nthreads = 3) == normal_fill!(rng, even; nthreads = 1)
        @test even == (A = similar(even); normal_fill!(rng, A; nthreads = 3); A)
    end
end

@testset "normals and exponentials: accuracy within Appendix A tolerances" begin
    # Each value against the formula in BigFloat on the same uniforms.
    tolerance(::Type{Float64}, z) = 1e-12 * abs(z) + 1e-15
    tolerance(::Type{Float32}, z) = 16 * eps(Float32(abs(z))) + 1e-6
    for T in (Float32, Float64)
        u = Vector{T}(undef, 2^14)
        rng = Tandem8x32(21)
        rand_fill!(rng, u)
        z = similar(u)
        exponential_fill!(rng, z)
        @test all(abs(z[i] + log1p(-big(u[i]))) <= tolerance(T, z[i]) for i in eachindex(u))
        normal_fill!(rng, z)
        reference = map(1:2:length(u)) do i
            r = sqrt(-2 * log1p(-big(u[i])))
            angle = 2 * big(pi) * u[i+1]
            return (r * cos(angle), r * sin(angle))
        end
        @test all(abs(z[2j-1] - reference[j][1]) <= tolerance(T, z[2j-1]) &&
                  abs(z[2j] - reference[j][2]) <= tolerance(T, z[2j]) for j in eachindex(reference))
    end
end

@testset "exponentials: Exp(1) moments and Kolmogorov-Smirnov on 1e7 draws" begin
    # Raw moments 1, 2, 6, 24 within 5 standard errors. The standard deviation of x^k is
    # sqrt((2k)! − (k!)^2). The KS bound 1.95/sqrt(n) is the 0.001 critical value.
    n = 10^7
    for T in (Float32, Float64)
        x = Vector{T}(undef, n)
        exponential_fill!(Tandem8x32(2026), x)
        for k = 1:4
            m = sum(Float64(v)^k for v in x) / n
            se = sqrt(factorial(2k) - factorial(k)^2) / sqrt(n)
            @test abs(m - factorial(k)) < 5se
        end
        sort!(x)
        d = maximum(eachindex(x)) do i
            F = -expm1(-Float64(x[i]))
            return max(i / n - F, F - (i - 1) / n)
        end
        @test d < 1.95 / sqrt(n)
    end
end

@testset "normals: N(0,1) moments and Kolmogorov-Smirnov on 1e7 draws" begin
    # Raw moments 0, 1, 0, 3 within 5 standard errors, with E z^2k = 1, 3, 15, 105. A pair is
    # standard bivariate normal when its squared radius is Exp(1/2) and its angle is uniform
    # and independent, so KS tests both margins of the 5e6 pairs at the 0.001 level.
    function ks_passes(x, F)
        sort!(x)
        m = length(x)
        d = maximum(i -> max(i / m - F(x[i]), F(x[i]) - (i - 1) / m), eachindex(x))
        return d < 1.95 / sqrt(m)
    end
    n = 10^7
    for T in (Float32, Float64)
        z = Vector{T}(undef, n)
        normal_fill!(Tandem8x32(2027), z)
        for (k, μ, μ2k) in ((1, 0, 1), (2, 1, 3), (3, 0, 15), (4, 3, 105))
            m = sum(Float64(v)^k for v in z) / n
            @test abs(m - μ) < 5 * sqrt((μ2k - μ^2) / n)
        end
        x, y = Float64.(z[1:2:end]), Float64.(z[2:2:end])
        @test ks_passes(@.(x^2 + y^2), t -> -expm1(-t / 2))
        @test ks_passes(atan.(y, x), t -> (t + π) / 2π)
    end
end

@testset "Random API: bounded, normal, and exponential draws" begin
    rng = after_bool(13)
    st = Stateful(rng)
    @test rand(st, 1:6) == rand_next(rng, 1:6)[1]
    @test rand(st, [:a, :b, :c]) == (:a, :b, :c)[rand_next(rand_next(rng, 1:6)[2], 1:3)[1]]
    for (A, B, r) in ((Vector{Int}(undef, 99), Vector{Int}(undef, 99), -5:Int(2)^31),
                      (Vector{Int8}(undef, 9), Vector{Int8}(undef, 9), Int8(-3):Int8(3)))
        st = Stateful(rng)
        rand!(st, A, r)
        @test Tandem8x32(st) == rand_fill!(rng, B, r)
        @test A == B
    end
    for T in (Float32, Float64)
        st = Stateful(rng)
        @test randn(st, T) == normal_next(rng, T)[1]
        @test randexp(st, T) == exponential_next(normal_next(rng, T)[2], T)[1]
        A, B = Vector{T}(undef, 101), Vector{T}(undef, 101)
        st = Stateful(rng)
        randn!(st, A)
        @test Tandem8x32(st) == normal_fill!(rng, B)
        @test A == B
        st = Stateful(rng)
        randexp!(st, A)
        @test Tandem8x32(st) == exponential_fill!(rng, B)
        @test A == B
    end
end
