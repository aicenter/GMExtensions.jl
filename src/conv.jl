"""
    conv_encoder(xsize, zsize, kernelsizes, channels, scalings[; activation, densedims])

Constructs a convolutional encoder.

# Arguments
- `xsize`: size of input - (h,w,c)
- `zsize`: latent space dimension
- `kernelsizes`: kernelsizes for conv. layers (only odd numbers are allowed)
- `channels`: channel numbers
- `scalings`: scalings vector
- `activation`: default relu
- `densedims`: if set, more than one dense layers are used 

# Example
```julia-repl
julia> encoder = conv_encoder((64, 48, 1), 2, (3, 5, 5), (2, 4, 8), (2, 2, 2), densedims = (256))
Chain(Conv((3, 3), 1=>2, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), Conv((5, 5), 2=>4, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), Conv((5, 5), 4=>8, relu), MaxPool((2, 2), pad = (0, 0, 0, 0), stride = (2, 2)), #44, Dense(384, 2))

julia> encoder(randn(Float32, 64, 48, 1, 2))
2×2 Array{Float32,2}:
  0.247844   0.133781
 -0.605763  -0.494911
```
"""
function conv_encoder(xsize::Union{Tuple, Vector}, zsize::Int, kernelsizes::Union{Tuple, Vector}, 
    channels::Union{Tuple, Vector}, scalings::Union{Tuple, Vector}; 
    activation = relu, densedims::Union{Tuple, Vector} = [])
    nconv = length(kernelsizes)
    (nconv == length(channels) == length(scalings)) ? nothing : error("incompatible input dimensions")
    (length(xsize) == 3) ? nothing : error("xsize must be (h, w, c)")
    # also check that kernelsizes are all odd numbers
    (all(kernelsizes .% 2 .== 1)) ? nothing : error("not implemented for even kernelsizes")

    # initialize some stuff
    cins = vcat(xsize[3], channels[1:end-1]...) # channels in
    couts = channels # channels out
    ho = xsize[1]/(reduce(*, scalings)) # height before reshaping
    wo = xsize[2]/(reduce(*, scalings)) # width before reshaping
    (ho == floor(Int, ho)) ? ho = floor(Int, ho) : error("your input size and scaling is not compatible")
    (wo == floor(Int, wo)) ? wo = floor(Int, wo) : error("your input size and scaling is not compatible")
    din = ho*wo*channels[end]

    # now build a vector of layers to be used later
    layers = Array{Any,1}()
    # first add the convolutional and maxpooling layers
    for (k, ci, co, s) in zip(kernelsizes, cins, couts, scalings)
        pad = Int((k-1)/2)
        # paddding so that input and output size are same
        push!(layers, Conv((k,k), ci=>co, activation; pad = (pad, pad))) 
        push!(layers, MaxPool((s,s)))
    end

    # reshape
    push!(layers, x -> reshape(x, din, :))

    # and dense layers
    ndense = length(densedims)
    dins = vcat(din, densedims...)
    douts = vcat(densedims..., zsize)
    dacts = vcat([activation for _ in 1:ndense]..., identity)
    for (_di, _do, _da) in zip(dins, douts, dacts)
        push!(layers, Dense(_di, _do, _da))
    end

    Flux.Chain(layers...)
end

"""
    conv_decoder(xsize, zsize, kernelsizes, channels, scalings[; activation, densedims, vec_output,
        vec_output_dim])

Constructs a convolutional encoder.

# Arguments
- `xsize`: size of input - (h,w,c)
- `zsize`: latent space dimension
- `kernelsizes`: kernelsizes for conv. layers (only odd numbers are allowed)
- `channels`: channel numbers
- `scalings`: scalings vector
- `activation`: default relu
- `densedims`: if set, more than one dense layers are used 
- `vec_output`: output is vectorized (default false)
- `vec_output_dim`: determine what the final size of the vectorized input should be (e.g. add extra dimensions for variance estimation)

# Example
```julia-repl
julia> decoder = conv_decoder((64, 48, 1), 2, (5, 5, 3), (8, 4, 2), (2, 2, 2); densedims = (256))
Chain(Dense(2, 256, relu), Dense(256, 384, relu), #19, ConvTranspose((2, 2), 8=>8, relu), Conv((5, 5), 8=>4, relu), ConvTranspose((2, 2), 4=>4, relu), Conv((5, 5), 4=>2, relu), ConvTranspose((2, 2), 2=>2, relu), Conv((3, 3), 2=>1))

julia> y = decoder(randn(Float32, 2, 2));

julia> size(y)
(64, 48, 1, 2)
```
"""
function conv_decoder(xsize::Union{Tuple, Vector}, zsize::Int, kernelsizes::Union{Tuple, Vector}, 
    channels::Union{Tuple, Vector}, scalings::Union{Tuple, Vector}; 
    activation = relu, densedims::Union{Tuple, Vector} = [], 
    vec_output = false, vec_output_dim = nothing)
    nconv = length(kernelsizes)
    (nconv == length(channels) == length(scalings)) ? nothing : error("incompatible input dimensions")
    (length(xsize) == 3) ? nothing : error("xsize must be (h, w, c)")
    # also check that kernelsizes are all odd numbers
    (all(kernelsizes .% 2 .== 1)) ? nothing : error("not implemented for even kernelsizes")

    # initialize some stuff
    cins = channels # channels in
    couts = vcat(channels[2:end]..., xsize[3]) # channels out
    ho = xsize[1]/(reduce(*, scalings)) # height after reshaping
    wo = xsize[2]/(reduce(*, scalings)) # width after reshaping
    (ho == floor(Int, ho)) ? ho = floor(Int, ho) : error("your input size and scaling is not compatible")
    (wo == floor(Int, wo)) ? wo = floor(Int, wo) : error("your input size and scaling is not compatible")
    dout = ho*wo*channels[1]

    # now build a vector of layers to be used later
    layers = Array{Any,1}()

    # start with dense layers
    ndense = length(densedims)
    dins = vcat(zsize, densedims...)
    douts = vcat(densedims..., dout)
    for (_di, _do) in zip(dins, douts)
        push!(layers, Dense(_di, _do, activation))
    end

    # reshape
    push!(layers, x -> reshape(x, ho, wo, channels[1], :))

    # add the transpose and convolutional layers
    acts = vcat([activation for _ in 1:nconv-1]..., identity) 
    for (k, ci, co, s, act) in zip(kernelsizes, cins, couts, scalings, acts)
        pad = Int((k-1)/2)
        # use convtranspose for upscaling - there are other posibilities, however this seems to be ok
        push!(layers, ConvTranspose((s,s), ci=>ci, activation, stride = (s,s))) 
        push!(layers, Conv((k,k), ci=>co, act; pad = (pad, pad)))
    end

    # if you want vectorized output
    if vec_output
        push!(layers, x -> reshape(x, reduce(*, xsize), :))
        # if you want the final vector to be a different size than the output 
        # (e.g. you want extra dimensions for variance estimation)
        (vec_output_dim == nothing) ? nothing : push!(layers, Dense(reduce(*, xsize), vec_output_dim))
    end
    
    Flux.Chain(layers...)
end
