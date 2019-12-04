module GMExtensions

    using DrWatson
    using ProgressMeter
    using ValueHistories

    using Plots
    using StatsPlots

    using Flux
    using ConditionalDists
    using IPMeasures
    using GenerativeModels

    using GenerativeModels: AbstractGM

    include("layers.jl")
    include("plotrecipes.jl")
    include("callbacks.jl")
    # include("train.jl")
    # include("reidentify.jl")
end
