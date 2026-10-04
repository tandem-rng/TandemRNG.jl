pushfirst!(LOAD_PATH, normpath(joinpath(@__DIR__, "..")))

using Documenter
using TandemRNG
using Random

DocMeta.setdocmeta!(TandemRNG, :DocTestSetup, :(using TandemRNG); recursive = true)

makedocs(;
    modules = [TandemRNG],
    authors = "Jessica Cox <jmcox@posteo.de>",
    sitename = "TandemRNG.jl",
    remotes = nothing,
    doctest = true,
    linkcheck = true,
    checkdocs = :exports,
    warnonly = [:linkcheck],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        repolink = "https://github.com/tandem-rng/TandemRNG.jl",
        canonical = "https://bjmcox.github.io/TandemRNG.jl/",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Streams and reproducibility" => "streams.md",
        "Derived draws" => "derived.md",
        "Devices" => "devices.md",
        "Integrations" => "integrations.md",
        "Performance" => "performance.md",
        "Design and validation" => "design.md",
        "API reference" => "api.md",
    ],
)
