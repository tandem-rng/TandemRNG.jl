# Weighted choice of Appendix C of the specification against the spec's conformance file
# conformance/choice.json, and the consumption and decomposition rules of the fills. It uses
# the fixture readers and `scalar_draws` of derived.jl.

using JSON

hex(T, s) = parse(T, s; base = 16)

@testset "choice: conformance cases" begin
    # Byte identical to tandem-spec 2a4bd08 conformance/choice.json.
    @test fixture_sha256("choice.json") ==
          "53e54b9bcd73a7ac9b355e9d09b74b922f6bbea73c825c9b098ff7a144435801"
    cases = JSON.parsefile(joinpath(FIXTURES, "choice.json"))["cases"]
    @test count(c -> haskey(c, "cut"), cases) == 5
    for c in cases
        t = ChoiceTable(reinterpret.(Float64, hex.(UInt64, c["weights"])))
        @test t.capacity == hex(UInt64, c["capacity"])
        if haskey(c, "cut")
            @test t.cut == hex.(UInt64, c["cut"])
            @test t.alias == hex.(UInt32, c["alias"]) .+ 1
        end
        rng = Tandem8x32{c["K"]}(Tuple(hex.(UInt32, c["key"])), c["start"])
        want = Int.(hex.(UInt32, c["values"])) .+ 1
        stop = get(c, "end", 64 * cld(c["start"], 64) + 64 * c["n"])
        A = Vector{Int}(undef, c["n"])
        @test rngposition(choice_fill!(rng, A, t)) == stop
        @test A == want
        st = Stateful(rng)
        B = Vector{UInt32}(undef, c["n"])
        rand!(st, B, t)
        @test B == want
        @test rngposition(Tandem8x32(st)) == stop
        isempty(want) && continue
        got, next = scalar_draws(r -> choice_next(r, t), rng, length(want))
        @test got == want
        @test rngposition(next) == stop
        st = Stateful(rng)
        @test [rand(st, t) for _ in want] == want
        # Pieces filled in order on one generator equal the whole fill.
        for cut in filter(<(length(want)), [1, 7, 20, 21, length(want) - 1])
            head, tail = Vector{Int}(undef, cut), Vector{Int}(undef, length(want) - cut)
            choice_fill!(choice_fill!(rng, head, t), tail, t)
            @test vcat(head, tail) == want
        end
    end
end

@testset "choice: threads and rejected weights" begin
    t = ChoiceTable([1, 2, 3, 4])
    rng = Tandem8x32(rngkey(Tandem8x32(42)), 33)
    # 3000 draws span 24 groups at K = 32.
    A = Vector{Int}(undef, 3000)
    @test choice_fill!(rng, A, t; nthreads = 3) == choice_fill!(rng, similar(A), t; nthreads = 1)
    @test A == (B = similar(A); choice_fill!(rng, B, t; nthreads = 1); B)
    for w in (Float64[], [1, -1], [1, NaN], [1, Inf], [0.0, -0.0])
        @test_throws ArgumentError ChoiceTable(w)
    end
end

@testset "choice: chi-square against the weights on 1e6 draws" begin
    # A zero weight never appears. Nine positive weights leave 8 degrees of freedom: the
    # 0.0005 and 0.9995 quantiles are 0.71 and 27.87.
    w = [0.5, 3, 0, 1, 7, 2.25, 0.1, 4, 1, 6]
    n = 10^6
    A = Vector{Int32}(undef, n)
    choice_fill!(Tandem8x32(2028), A, ChoiceTable(w))
    counts = [count(==(i), A) for i in eachindex(w)]
    @test counts[3] == 0
    expected = n .* w ./ sum(w)
    chi2 = sum((counts[i] - expected[i])^2 / expected[i] for i in eachindex(w) if w[i] > 0)
    @test 0.71 < chi2 < 27.87
end
