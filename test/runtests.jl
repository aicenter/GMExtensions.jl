using Test
using Logging
using ProgressMeter
using ValueHistories
using Plots
using Flux
using Distributions
using DistributionsAD
using ConditionalDists
using IPMeasures
using GenerativeModels
using GMExtensions

# for testing of parameter change in training
get_params(model) =  map(copy, collect(params(model)))
param_change(frozen_params, model) = 
	map(x-> x[1] != x[2], zip(frozen_params, collect(params(model))))

include("saveload.jl")
include("layers.jl")
include("plotrecipes.jl")
include("callbacks.jl")
#include("reidentify.jl")
include("conv.jl")
