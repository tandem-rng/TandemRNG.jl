# GPU benchmarks on one CUDA device: the fill against cuRAND's Philox4x32-10 and CUDA.jl's
# native generators;
# the `fill_lanes!` kernel over workgroup sizes; scalar Float64 chains in a kernel; and
# in-kernel draws against Random123's Philox4x32-10 core in the same kernel shape. Random123
# has no CUDA array-fill API, so its chains and core comparison use its stateless core.
# Run from an environment with CUDA, Random123, Random, and TandemRNG.
# KernelAbstractions comes through the extension, so the environment needs no extra dependency.
#
# The GPUs are shared. `run` refuses a busy device unless forced, and prints the idle check
# from before and after the timings next to the table, so a figure carries its evidence.
module GPUBench

using CUDA
import Random123
using Random
using TandemRNG
using TandemRNG: O4, DOMAIN_STREAM, AUX_STREAM, ROW_BITS, draw_bits, seed, step

const Ext = Base.get_extension(TandemRNG, :TandemRNGKernelAbstractionsExt)
const KA = Ext.KernelAbstractions
using .KA: @kernel, @index

const TYPES = (Float32, UInt32, Float64, Bool)
# `CUDA.default_rng()` selects CUDA's GPUArrays generator. Its algorithm depends on the
# CUDA version. `native_rng()` selects the distinct kernel generator of the cuRAND package.
# The cuRAND library row uses its Philox4x32-10 generator, not its XORWOW default.
const GENERATORS = (
    "TandemRNG K=32",
    "cuRAND Philox4x32-10",
    "CUDA.jl native",
    "cuRAND NativeRNG",
)
const CHUNK = 32

function idle_check(gpu; samples = 5)
    readings = map(1:samples) do i
        i > 1 && sleep(0.5)
        split(
            readchomp(
                `nvidia-smi --query-gpu=uuid,name,utilization.gpu --format=csv,noheader -i $gpu`,
            ),
            ", ",
        )
    end
    uuid, name = readings[1]
    utilization = [parse(Int, strip(r[3], [' ', '%'])) for r in readings]
    apps = eachline(
        `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`,
    )
    others = String[]
    for (g, pid, mem) in (split(l, ", ") for l in apps)
        g == uuid && parse(Int, pid) != getpid() && push!(others, "pid $pid ($mem)")
    end
    return (; name = String(name), others, utilization)
end

idle(check) = isempty(check.others) && all(==(0), check.utilization)

function describe(check)
    others = isempty(check.others) ? "none" : join(check.others, "; ")
    return "$(check.name), other processes: $others, utilization %: $(check.utilization)"
end

# Minimum over `reps` event timings after `warm` seconds of the same fill. The A100's SM
# clock idles at 765 MHz and reaches 1410 MHz only under sustained load, so a short burst
# after a pause (a kernel compile, an allocation) measured 30-60 % low and varied between
# passes. The synchronize drains fills that the warm loop queued without waiting.
function best_seconds(f, reps; warm = 0.5)
    f()
    CUDA.synchronize()
    t0 = time()
    while time() - t0 < warm
        f()
    end
    CUDA.synchronize()
    return minimum(Float64(CUDA.@elapsed f()) for _ = 1:reps)
end

gibps(bytes, seconds) = bytes / 2^30 / seconds

fmt(x) = x === nothing ? "–" : string(round(Int, x))

function markdown(io, header, rows)
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", repeat(" --- |", length(header)))
    for row in rows
        println(io, "| ", join(row, " | "), " |")
    end
end

function generators(::Type{T}, A) where {T}
    tandem = TandemRNG.MLDataDevices.CUDADevice()(Tandem8x32{CHUNK}(1))
    native = CUDA.default_rng()
    kernel_rng = CUDA.CURAND.native_rng()
    philox = CUDA.CURAND.LibraryRNG(CUDA.CURAND.CURAND_RNG_PSEUDO_PHILOX4_32_10)
    fills = Dict{String,Any}(
        "TandemRNG K=32" => () -> rand_fill!(tandem, A),
        "CUDA.jl native" => () -> Random.rand!(native, A),
        "cuRAND NativeRNG" => () -> Random.rand!(kernel_rng, A),
    )
    # The cuRAND library has no Bool generator.
    T === Bool || (fills["cuRAND Philox4x32-10"] = () -> Random.rand!(philox, A))
    return fills
end

function compare_gpu(io::IO; gpu, sizes = (1 << 20, 1 << 27), passes = 3, reps = 30)
    passes > 0 && reps > 0 || throw(ArgumentError("passes and reps must be positive"))
    !isempty(sizes) && all(n -> n > 0, sizes) ||
        throw(ArgumentError("sizes must contain positive element counts"))
    before = idle_check(gpu)
    idle(before) || error("GPU $gpu is not idle: $(describe(before))")
    CUDA.device!(gpu)
    CUDA.allowscalar(false)
    println(io, "# before: ", describe(before))
    println(io, "pass\ttype\telements\tgenerator\tseconds\tGiB_s\thost_allocated_bytes")
    for T in TYPES, n in sizes
        A = CuVector{T}(undef, n)
        fills = generators(T, A)
        # Validate the public fill before timing large fills.
        small = CuVector{T}(undef, 1025)
        rng = Tandem8x32{CHUNK}(1)
        expected = Vector{T}(undef, length(small))
        after = rand_fill!(rng, expected; nthreads = 1)
        bound = TandemRNG.MLDataDevices.CUDADevice()(rng)
        native = rand_fill!(bound, small)
        Array(small) == expected || error("GPU native fill differs from CPU")
        TandemRNG.MLDataDevices.CPUDevice()(native) == after ||
            error("GPU end state differs")
        for pass = 1:passes, name in (isodd(pass) ? GENERATORS : reverse(GENERATORS))
            haskey(fills, name) || continue
            fill = fills[name]
            seconds = best_seconds(fill, reps)
            allocated = @allocated begin
                fill()
                CUDA.synchronize()
            end
            println(
                io,
                join(
                    (pass, T, n, name, seconds, gibps(sizeof(T) * n, seconds), allocated),
                    '\t',
                ),
            )
            flush(io)
        end
        CUDA.unsafe_free!(small)
        CUDA.unsafe_free!(A)
    end
    after = idle_check(gpu)
    println(io, "# after: ", describe(after))
    # Utilization can still include this run because nvidia-smi samples an interval.
    isempty(after.others) || error("Another GPU process appeared during the run")
    return nothing
end

@kernel function public_chain!(output, states, draw, count)
    i = @index(Global)
    rng = states[i]
    acc = 0.0
    for _ = 1:count
        x, rng = draw(rng, Float64)
        acc += x
    end
    output[i] = acc
end

# Random123's Philox4x32-10 core as an immutable chain: each block gives two Float64 draws of
# 53 bits, then the counter advances. Random123's own generators are mutable and host-only.
struct PhiloxChain
    key::NTuple{2,UInt32}
    counter::UInt64
    block::O4
    half::Bool
end

@inline philox_block(key, n::UInt64) =
    Random123.philox(key, (n % UInt32, (n >> 32) % UInt32, UInt32(0), UInt32(0)), Val(10))

function PhiloxChain(i::Integer)
    key = (i % UInt32, 0x9e3779b9)
    return PhiloxChain(key, UInt64(0), philox_block(key, UInt64(0)), false)
end

@inline function philox_next(rng::PhiloxChain, ::Type{Float64})
    b = rng.block
    raw = rng.half ? UInt64(b[3]) | UInt64(b[4]) << 32 : UInt64(b[1]) | UInt64(b[2]) << 32
    x = Float64(raw >> 11) * 0x1p-53
    rng.half || return x, PhiloxChain(rng.key, rng.counter, b, true)
    n = rng.counter + UInt64(1)
    return x, PhiloxChain(rng.key, n, philox_block(rng.key, n), false)
end

function compare_public_draws(io::IO; gpu, n = 1 << 16, count = 1024, passes = 3, reps = 30)
    n > 0 && count > 0 && passes > 0 && reps > 0 ||
        throw(ArgumentError("counts, passes, and reps must be positive"))
    before = idle_check(gpu)
    idle(before) || error("GPU $gpu is not idle: $(describe(before))")
    CUDA.device!(gpu)
    CUDA.allowscalar(false)
    device = TandemRNG.MLDataDevices.CUDADevice()
    cases = (
        ("Tandem native", i -> device(Tandem8x32{CHUNK}(i)), TandemRNG.rand_next),
        ("Random123 Philox4x32", PhiloxChain, philox_next),
    )
    println(io, "# before: ", describe(before))
    println(io, "pass\tgenerator\tchains\tdraws_per_chain\tseconds\tGiB_s_generated")
    fixtures = map(cases) do (name, constructor, draw)
        cpu = [constructor(i) for i = 1:n]
        states = CuArray(cpu)
        output = CuVector{Float64}(undef, n)
        kernel = public_chain!(CUDABackend(), 256)
        call = () -> kernel(output, states, draw, count; ndrange = n)
        call()
        actual = Array(output)
        for i = 1:min(n, 32)
            rng, acc = cpu[i], 0.0
            for _ = 1:count
                x, rng = draw(rng, Float64)
                acc += x
            end
            actual[i] == acc || error("$name GPU scalar chain differs from CPU")
        end
        (; name, call, states, output)
    end
    for pass = 1:passes, fixture in (isodd(pass) ? fixtures : reverse(fixtures))
        seconds = best_seconds(fixture.call, reps)
        println(
            io,
            join(
                (
                    pass,
                    fixture.name,
                    n,
                    count,
                    seconds,
                    gibps(n * count * sizeof(Float64), seconds),
                ),
                '\t',
            ),
        )
        flush(io)
    end
    for fixture in fixtures
        CUDA.unsafe_free!(fixture.states)
        CUDA.unsafe_free!(fixture.output)
    end
    after = idle_check(gpu)
    println(io, "# after: ", describe(after))
    isempty(after.others) || error("Another GPU process appeared during the run")
    return nothing
end

function run(; gpu, n = 1 << 27, reps = 30, force = false, io::IO = stdout)
    before = idle_check(gpu)
    force ||
        idle(before) ||
        error("GPU $gpu is not idle: $(describe(before)). Pass force = true to run anyway.")
    CUDA.device!(gpu)
    println(
        io,
        "Julia $VERSION, CUDA $(pkgversion(CUDA)), Random123 $(pkgversion(Random123))",
    )
    println(
        io,
        "CUDA default: $(typeof(CUDA.default_rng())), cuRAND native: $(typeof(CUDA.CURAND.native_rng()))",
    )
    rows = map(TYPES) do T
        A = CuVector{T}(undef, n)
        fills = generators(T, A)
        rates = [
            haskey(fills, g) ? gibps(n * sizeof(T), best_seconds(fills[g], reps)) : nothing for g in GENERATORS
        ]
        (T, rates)
    end
    after = idle_check(gpu)
    markdown(io, ["type", GENERATORS...], [(string(T), fmt.(r)...) for (T, r) in rows])
    println(io, "GiB/s written, n = $n, min of $reps, GPU $gpu")
    println(io, "before: ", describe(before))
    println(io, "after:  ", describe(after))
    return (; rows, before, after)
end

# The extension kernel alone, aligned path, over workgroup sizes. The launch mirrors
# `TandemRNG.rand_fill!` in the extension.
function sweep(; gpu, n = 1 << 27, reps = 30, force = false, io::IO = stdout)
    before = idle_check(gpu)
    force || idle(before) || error("GPU $gpu is not idle: $(describe(before))")
    CUDA.device!(gpu)
    rng = Tandem8x32{CHUNK}(1)
    rows = []
    for T in (Float32, UInt32)
        A = CuVector{T}(undef, n)
        expected = Vector{T}(undef, n)
        rand_fill!(rng, expected; nthreads = 1)
        s = sizeof(T)
        nb = draw_bits(T)
        p0 = TandemRNG._align_up(rng.pos, nb)
        pend = p0 + UInt64(nb) * UInt64(n)
        bits_per_group = UInt64(ROW_BITS) * UInt64(CHUNK)
        g0 = p0 ÷ bits_per_group
        nchunks = Int((pend - 1) ÷ bits_per_group - g0 + 1) * 8
        for W in (64, 128, 256, 512, 1024)
            kernel = Ext.fill_lanes!(CUDABackend(), W)
            fill =
                () -> begin
                    kernel(
                        A,
                        rng.key,
                        g0 << 3,
                        p0,
                        pend,
                        Val(CHUNK),
                        Val(true);
                        ndrange = nchunks,
                    )
                end
            seconds = best_seconds(fill, reps)
            Array(A) == expected || error("fill_lanes! differs from the CPU stream")
            push!(rows, (string(T), W, fmt(gibps(n * s, seconds))))
        end
    end
    after = idle_check(gpu)
    markdown(io, ["type", "W", "GiB/s"], rows)
    println(io, "before: ", describe(before))
    println(io, "after:  ", describe(after))
    return rows
end

# --- in-kernel draws ----------------------------------------------------------------------
# C chunks per work-item, K steps each, every output word folded into one accumulator per
# chunk, one word stored per chunk. The store is negligible, so the time is the generator's
# arithmetic. Work-item i owns chunks C(i−1) to C(i−1) + C − 1 and advances their step
# chains in one loop, so the compiler sees C independent dependency chains per thread and
# C = 2 or 4 tells a latency-bound kernel from an issue-bound one. The chains live in
# tuples because `map` over a fixed-length tuple unrolls at compile time and keeps every
# state word in a register. The seed closure carries eight inlined steps, which is past the
# inliner's cost limit, so it needs `@inline`: as a call it went through local memory and
# kept an InexactError path for `UInt64(c - 1)` alive.

@inline fold(acc::UInt32, o::O4) = acc ⊻ o[1] ⊻ o[2] ⊻ o[3] ⊻ o[4]

@kernel function draws_tandem!(out, key::O4, ::Val{K}, ::Val{C}) where {K,C}
    i = @index(Global, Linear)
    base = UInt64(i - 1) * UInt64(C)
    chains = ntuple(
        @inline(c -> seed(key, base + UInt64(c - 1), DOMAIN_STREAM, AUX_STREAM)),
        Val(C),
    )
    accs = ntuple(_ -> UInt32(0), Val(C))
    for _ = 1:K
        chains = map(oh -> step(oh[1], oh[2]), chains)
        accs = map((acc, oh) -> fold(acc, oh[1]), accs, chains)
    end
    for c = 1:C
        @inbounds out[(i-1)*C+c] = accs[c]
    end
end

@inline external_philox(c::O4, k::NTuple{2,UInt32}) = Random123.philox(k, c, Val(10))

# Chunk j owns counters j·K to j·K + K − 1 in the low two counter words. Same chunk
# ownership and chain interleave as `draws_tandem!`.
@kernel function draws_philox!(
    out,
    key::NTuple{2,UInt32},
    block,
    ::Val{K},
    ::Val{C},
) where {K,C}
    i = @index(Global, Linear)
    base = UInt64(i - 1) * UInt64(C * K)
    counters = ntuple(c -> base + UInt64((c - 1) * K), Val(C))
    accs = ntuple(_ -> UInt32(0), Val(C))
    for _ = 1:K
        accs = map(
            (acc, n) -> fold(
                acc,
                block((n % UInt32, (n >> 32) % UInt32, UInt32(0), UInt32(0)), key),
            ),
            accs,
            counters,
        )
        counters = map(n -> n + UInt64(1), counters)
    end
    for c = 1:C
        @inbounds out[(i-1)*C+c] = accs[c]
    end
end

# Every chain count covers the same `n_chunks` chunks, so the rows compare equal work.
function draws(;
    gpu,
    n_chunks = 1 << 20,
    K = 32,
    reps = 30,
    chains = (1, 2, 4),
    force = false,
    io::IO = stdout,
)
    all(C -> n_chunks % C == 0, chains) || throw(
        ArgumentError(
            "n_chunks = $n_chunks is not a multiple of every chain count $chains",
        ),
    )
    before = idle_check(gpu)
    force || idle(before) || error("GPU $gpu is not idle: $(describe(before))")
    CUDA.device!(gpu)
    validate_draws(K, chains)
    out = CuVector{UInt32}(undef, n_chunks)
    key = Tandem8x32{K}(1).key
    bytes = n_chunks * K * 16
    rate(kernel, args, C) = gibps(
        bytes,
        best_seconds(
            () -> kernel(CUDABackend(), 256)(
                out,
                args...,
                Val(K),
                Val(C);
                ndrange = n_chunks ÷ C,
            ),
            reps,
        ),
    )
    tandem = [rate(draws_tandem!, (key,), C) for C in chains]
    k = (key[1], key[2])
    external = [rate(draws_philox!, (k, external_philox), C) for C in chains]
    after = idle_check(gpu)
    markdown(
        io,
        ["chains per thread", "TandemRNG K=$K", "Random123 Philox4x32"],
        [(string(C), fmt(t), fmt(r)) for (C, t, r) in zip(chains, tandem, external)],
    )
    println(
        io,
        "Core throughput, GiB/s generated, n_chunks = $n_chunks, K = $K, 16 bytes per step, min of $reps, GPU $gpu",
    )
    println(io, "before: ", describe(before))
    println(io, "after:  ", describe(after))
    return (; chains, tandem, external, before, after)
end

# Untimed checks cover every chain shape and compare GPU words with scalar CPU folds.
function validate_draws(K, chains; n = 1024)
    key = Tandem8x32{K}(1).key
    k = (key[1], key[2])
    tandem = Vector{UInt32}(undef, n)
    philox = similar(tandem)
    for j = 0:(n-1)
        o, h = seed(key, UInt64(j), DOMAIN_STREAM, AUX_STREAM)
        ta, pa = UInt32(0), UInt32(0)
        for s = 0:(K-1)
            o, h = step(o, h)
            ta = fold(ta, o)
            counter = UInt64(j * K + s)
            c = (counter % UInt32, (counter >> 32) % UInt32, UInt32(0), UInt32(0))
            pa = fold(pa, external_philox(c, k))
        end
        tandem[j+1], philox[j+1] = ta, pa
    end
    out = CuVector{UInt32}(undef, n)
    for C in chains
        n % C == 0 || throw(ArgumentError("validation size must divide every chain count"))
        draws_tandem!(CUDABackend(), 256)(out, key, Val(K), Val(C); ndrange = n ÷ C)
        Array(out) == tandem || error("Tandem GPU draw fold differs from CPU")
        draws_philox!(CUDABackend(), 256)(
            out,
            k,
            external_philox,
            Val(K),
            Val(C);
            ndrange = n ÷ C,
        )
        Array(out) == philox || error("Philox GPU draw fold differs from CPU")
    end
    return nothing
end

end
