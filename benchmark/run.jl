using Dates, SHA, TOML
import TandemRNG, Random123, BenchmarkTools

function options(args)
    length(args) >= 2 || throw(
        ArgumentError(
            "usage: run.jl cpu|gpu|latency OUTPUT [--sizes=N,N] [--passes=N] [--seconds=S] [--device=N]",
        ),
    )
    mode, output = args[1:2]
    mode in ("cpu", "gpu", "latency") || throw(ArgumentError("unknown mode: $mode"))
    ispath(output) && throw(ArgumentError("output path already exists: $output"))
    opts = Dict{String,String}()
    allowed =
        mode == "cpu" ? ("sizes", "passes", "seconds") :
        mode == "gpu" ? ("sizes", "passes", "device") : ("passes",)
    for arg in args[3:end]
        startswith(arg, "--") && occursin('=', arg) ||
            throw(ArgumentError("expected --name=value, got $arg"))
        key, value = split(arg[3:end], '='; limit = 2)
        key in allowed || throw(ArgumentError("unsupported $mode option: $key"))
        haskey(opts, key) && throw(ArgumentError("duplicate option: $key"))
        opts[key] = value
    end
    passes = parse(Int, get(opts, "passes", "3"))
    seconds = parse(Float64, get(opts, "seconds", "0.3"))
    device = parse(Int, get(opts, "device", "0"))
    default_sizes = mode == "gpu" ? "1048576,134217728" : "1024,1048576,16777216"
    sizes = parse.(Int, split(get(opts, "sizes", default_sizes), ','))
    passes > 0 && isfinite(seconds) && seconds > 0 && device >= 0 && all(>(0), sizes) ||
        throw(
            ArgumentError(
                "sizes, passes, and seconds must be positive; device must be nonnegative",
            ),
        )
    return (; mode, output = abspath(output), passes, seconds, device, sizes)
end

function source_hashes(modules)
    hashes = Dict{String,String}()
    for mod in modules
        root = pkgdir(mod)
        files = [joinpath(root, "Project.toml")]
        for dir in ("src", "ext")
            isdir(joinpath(root, dir)) || continue
            for (path, _, names) in walkdir(joinpath(root, dir)), name in names
                endswith(name, ".jl") && push!(files, joinpath(path, name))
            end
        end
        for path in files
            hashes[string(nameof(mod), "/", relpath(path, root))] =
                bytes2hex(sha256(read(path)))
        end
    end
    for name in ("run.jl", "benchmarks.jl", "gpu.jl", "first_use.jl")
        path = joinpath(@__DIR__, name)
        hashes["benchmark/$name"] = bytes2hex(sha256(read(path)))
    end
    return hashes
end

function run_benchmarks(config)
    mkpath(config.output)
    status = Dict{String,Any}("state" => "running", "started" => string(now(UTC)))
    save_status() = open(
        io -> TOML.print(io, status; sorted = true),
        joinpath(config.output, "status.toml"),
        "w",
    )
    save_status()
    try
        modules = (TandemRNG, Random123, BenchmarkTools)
        metadata = Dict{String,Any}(
            "julia" => string(VERSION),
            "machine" => Sys.MACHINE,
            "cpu" => Sys.CPU_NAME,
            "threads" => Threads.nthreads(),
            "word_size" => Sys.WORD_SIZE,
            "configuration" => Dict(string(k) => v for (k, v) in pairs(config)),
            "versions" =>
                Dict(string(nameof(m)) => string(pkgversion(m)) for m in modules),
            "source_hashes" => source_hashes(modules[1:3]),
        )
        open(
            io -> TOML.print(io, metadata; sorted = true),
            joinpath(config.output, "metadata.toml"),
            "w",
        )
        project = Base.active_project()
        cp(project, joinpath(config.output, "Project.toml"))
        manifest = joinpath(dirname(project), "Manifest.toml")
        isfile(manifest) && cp(manifest, joinpath(config.output, "Manifest.toml"))
        if config.mode == "cpu"
            open(joinpath(config.output, "cpu.tsv"), "w") do io
                compare_cpu(
                    io;
                    sizes = config.sizes,
                    passes = config.passes,
                    seconds = config.seconds,
                )
            end
        elseif config.mode == "gpu"
            metadata["versions"]["CUDA"] = string(pkgversion(GPUBench.CUDA))
            open(
                io -> TOML.print(io, metadata; sorted = true),
                joinpath(config.output, "metadata.toml"),
                "w",
            )
            open(joinpath(config.output, "gpu-fill.tsv"), "w") do io
                GPUBench.compare_gpu(
                    io;
                    gpu = config.device,
                    sizes = config.sizes,
                    passes = config.passes,
                )
            end
            open(joinpath(config.output, "gpu-public-draws.tsv"), "w") do io
                GPUBench.compare_public_draws(
                    io;
                    gpu = config.device,
                    passes = config.passes,
                )
            end
            open(joinpath(config.output, "gpu-core.md"), "w") do io
                GPUBench.draws(; gpu = config.device, chains = (1,), io)
            end
        else
            for pass = 1:config.passes,
                engine in ("tandem", "random123-32", "random123-64", "xoshiro")

                path = joinpath(config.output, "$engine-$pass.tsv")
                cmd = `$(Base.julia_cmd()) --startup-file=no --threads=2 --gcthreads=1 --project=$(dirname(project)) $(joinpath(@__DIR__, "first_use.jl")) $engine`
                open(path, "w") do io
                    run(pipeline(cmd; stdout = io, stderr = io))
                end
            end
        end
        status["state"] = "completed"
    catch err
        status["state"] = "failed"
        status["error"] = sprint(showerror, err)
        rethrow()
    finally
        status["finished"] = string(now(UTC))
        save_status()
    end
    return config.output
end

if abspath(PROGRAM_FILE) == @__FILE__
    config = options(ARGS)
    if config.mode == "cpu"
        include(joinpath(@__DIR__, "benchmarks.jl"))
    elseif config.mode == "gpu"
        include(joinpath(@__DIR__, "gpu.jl"))
    end
    run_benchmarks(config)
end
