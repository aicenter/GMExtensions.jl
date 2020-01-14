export Bias
export LinearMap
export SplitLayer
export CatLayer

struct Bias{T,F}
    b::T
    σ::F
end

Bias(len::Int, σ=identity; init=zeros) = Bias(init(Float32,len), σ)
(m::Bias)() = m.σ.(m.b)
(m::Bias)(x::AbstractArray) = m()
Flux.@functor Bias


struct LinearMap
    W::AbstractArray
end

LinearMap(in::Int, out::Int, initW=Flux.glorot_uniform) = LinearMap(initW(out, in))
(m::LinearMap)(x::AbstractArray) = m.W * x
Flux.@functor LinearMap

"""
    SplitLayer(layers::Tuple)

Splits input vector with into mulitple outputs e.g.:
x -> (layer[1](x), ..., layer[end](x))
"""
struct SplitLayer
    layers::Tuple
end

function SplitLayer(input::Int, outputs::Array{Int,1}, act=identity)
    layers = []
    for out in outputs
        push!(layers, Dense(input, out, act))
    end
    SplitLayer(Tuple(layers))
end

function (m::SplitLayer)(x)
    Tuple(layer(x) for layer in m.layers)
end

Flux.@functor SplitLayer


"""
    CatLayer(layers::Tuple)

Concatenates output of multiple layers e.g.:
x -> vcat(layer[1](x), ..., layer[end](x))
"""
struct CatLayer
    layers::Tuple
end

function CatLayer(layers...)
    CatLayer(layers)
end

function (m::CatLayer)(x)
    y = [layer(x) for layer in m.layers]
    vcat(y...)
end

Flux.@functor CatLayer
