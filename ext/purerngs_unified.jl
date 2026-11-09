# Reuse PureRNGs' storage, alias, and reconstruction rules for the external engine.
for (generate, next) in
    ((:randgen!!, :rand_next), (:randngen!!, :randn_next), (:randexpgen!!, :randexp_next))
    @eval @inline PureRNGs.$generate(rng::Generator, args...; kwargs...) =
        PureRNGs.$next(rng, args...; kwargs...)
end

@inline PureRNGs.randgen!!(
    rng::Generator,
    source,
    dims::Tuple{Int,Vararg{Int}};
    kwargs...,
) = PureRNGs.randgen!!(rng, source, dims...; kwargs...)

@inline PureRNGs._randset_fill!!(rng::Generator, source, destination; kwargs...) =
    PureRNGs.rand_next!(rng, source, destination; kwargs...)
@inline PureRNGs._randset_fill!!(rng::Generator, ::Type, destination; kwargs...) =
    PureRNGs.randuset!!(rng, destination; kwargs...)

for (set, next!) in (
    (:randuset!!, :rand_next!),
    (:randnset!!, :randn_next!),
    (:randexpset!!, :randexp_next!),
)
    @eval @inline PureRNGs._primitive_set!!(
        ::typeof(PureRNGs.$set),
        rng::Generator,
        destination;
        kwargs...,
    ) = PureRNGs.$next!(rng, destination; kwargs...)
end
