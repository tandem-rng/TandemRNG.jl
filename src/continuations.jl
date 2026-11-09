"""
    StatefulRNG(rng::Tandem8x32)
    StatefulRNG(seed::Integer)
    StatefulRNG{K}(seed::Integer)

Alias for [`Stateful`](@ref), retaining Tandem's native sampling algorithms.
"""
const StatefulRNG = Stateful

"""
    rand_next!(rng, destination; threaded::Bool = false) -> (destination, rng′)

Fill `destination` through [`rand_fill!`](@ref) and return it with the advanced generator.
The returned destination is the supplied object. `threaded = true` uses up to
`Threads.nthreads()` CPU tasks; the default uses one. Values and draw consumption follow
`rand_fill!`, including when a loaded backend supplies the fill.
"""
function rand_next!(rng::_UniformRNG, destination::AbstractArray; threaded::Bool = false)
    next_rng = rand_fill!(rng, destination; nthreads = threaded ? Threads.nthreads() : 1)
    return destination, next_rng
end

"""
    randn_next(rng, T = Float64) -> (z, rng′)
    randn_next(rng, [T,] dims::Dims; threaded::Bool = false) -> (A, rng′)
    randn_next(rng, [T,] n::Integer, dims::Integer...; threaded::Bool = false)

Draw native standard normals through [`normal_next`](@ref), or allocate and fill an array
through [`randn_next!`](@ref). `T` is `Float32` or `Float64`; the default is `Float64`.
Allocating forms use the generator's device binding and the native CPU normal fill.
The scalar and fill algorithms retain their respective Appendix A consumption rules.
"""
randn_next(rng::Tandem8x32, ::Type{T} = Float64) where {T<:NormalTypes} =
    normal_next(rng, T)

function randn_next(
    rng::Tandem8x32,
    ::Type{T},
    dims::Dims;
    threaded::Bool = false,
) where {T<:NormalTypes}
    destination = _allocate_draw_array(rng, T, dims)
    return randn_next!(rng, destination; threaded)
end

randn_next(rng::Tandem8x32, dims::Dims; threaded::Bool = false) =
    randn_next(rng, Float64, dims; threaded)
randn_next(
    rng::Tandem8x32,
    ::Type{T},
    n::Integer,
    dims::Integer...;
    threaded::Bool = false,
) where {T<:NormalTypes} = randn_next(rng, T, Int.((n, dims...)); threaded)
randn_next(rng::Tandem8x32, n::Integer, dims::Integer...; threaded::Bool = false) =
    randn_next(rng, Float64, Int.((n, dims...)); threaded)

"""
    randn_next!(rng, destination; threaded::Bool = false) -> (destination, rng′)

Fill a `Float32` or `Float64` CPU destination through [`normal_fill!`](@ref) and return
the same destination with the advanced generator. `threaded = true` uses up to
`Threads.nthreads()` tasks; the default uses one. Values and draw consumption follow
`normal_fill!` exactly.
"""
function randn_next!(
    rng::Tandem8x32,
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:NormalTypes}
    next_rng = normal_fill!(rng, destination; nthreads = threaded ? Threads.nthreads() : 1)
    return destination, next_rng
end

"""
    randexp_next(rng, T = Float64) -> (e, rng′)
    randexp_next(rng, [T,] dims::Dims; threaded::Bool = false) -> (A, rng′)
    randexp_next(rng, [T,] n::Integer, dims::Integer...; threaded::Bool = false)

Draw native standard exponentials through [`exponential_next`](@ref), or allocate and fill
an array through [`randexp_next!`](@ref). `T` is `Float32` or `Float64`; the default is
`Float64`. Allocating forms use the generator's device binding and the native CPU
exponential fill. Values and draw consumption retain the Appendix A rules.
"""
randexp_next(rng::Tandem8x32, ::Type{T} = Float64) where {T<:NormalTypes} =
    exponential_next(rng, T)

function randexp_next(
    rng::Tandem8x32,
    ::Type{T},
    dims::Dims;
    threaded::Bool = false,
) where {T<:NormalTypes}
    destination = _allocate_draw_array(rng, T, dims)
    return randexp_next!(rng, destination; threaded)
end

randexp_next(rng::Tandem8x32, dims::Dims; threaded::Bool = false) =
    randexp_next(rng, Float64, dims; threaded)
randexp_next(
    rng::Tandem8x32,
    ::Type{T},
    n::Integer,
    dims::Integer...;
    threaded::Bool = false,
) where {T<:NormalTypes} = randexp_next(rng, T, Int.((n, dims...)); threaded)
randexp_next(rng::Tandem8x32, n::Integer, dims::Integer...; threaded::Bool = false) =
    randexp_next(rng, Float64, Int.((n, dims...)); threaded)

"""
    randexp_next!(rng, destination; threaded::Bool = false) -> (destination, rng′)

Fill a `Float32` or `Float64` CPU destination through [`exponential_fill!`](@ref) and
return the same destination with the advanced generator. `threaded = true` uses up to
`Threads.nthreads()` tasks; the default uses one. Values and draw consumption follow
`exponential_fill!` exactly.
"""
function randexp_next!(
    rng::Tandem8x32,
    destination::AbstractArray{T};
    threaded::Bool = false,
) where {T<:NormalTypes}
    next_rng =
        exponential_fill!(rng, destination; nthreads = threaded ? Threads.nthreads() : 1)
    return destination, next_rng
end
