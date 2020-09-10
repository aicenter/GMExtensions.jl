module GMExtensions

    using Statistics
    using DrWatson
    using ProgressMeter
    using ValueHistories
    using BSON

    using Plots
    using StatsPlots

    using Flux
    using Zygote
    using ConditionalDists
    using IPMeasures
    using GenerativeModels

    using GenerativeModels: AbstractVAE, AbstractGM, ACD

    export save_checkpoint, load_checkpoint

    include("saveload.jl")
    include("layers.jl")
    include("plotrecipes.jl")
    include("callbacks.jl")
    # include("train.jl")
    include("reidentify.jl")
    include("conv.jl")
end
