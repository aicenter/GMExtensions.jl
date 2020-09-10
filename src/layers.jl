export LinearMap
export SplitLayer
export CatLayer

struct LinearMap
    W::AbstractArray
end

LinearMap(in::Int, out::Int, initW=Flux.glorot_uniform) = LinearMap(initW(out, in))
(m::LinearMap)(x::AbstractArray) = m.W * x
Flux.@functor LinearMap

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
