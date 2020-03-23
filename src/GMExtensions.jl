module GMExtensions

    using Statistics
    using DrWatson
    using ProgressMeter
    using ValueHistories

    using Plots
    using StatsPlots

    using Flux
    using Zygote
    using ConditionalDists
    using IPMeasures
    using GenerativeModels

    using GenerativeModels: AbstractVAE, AbstractGM, ACD

    include("layers.jl")
    include("plotrecipes.jl")
    include("callbacks.jl")
    # include("train.jl")
    include("reidentify.jl")
    include("conv.jl")
end
