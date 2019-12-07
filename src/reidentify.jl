export reidentify
export relevance_mask

"""
    reidentify(z::AbstractMatrix, zmask::AbstractMatrix,
               u::AbstractMatrix, decoder::AbstractCPDF, opt; N=200)

Optimize MSE(decoder(z) - u) where only the values of z that are specified by
zmask are allowed to change.

# Arguments
* `z`: typically latent variables typically sampled from the encoder of an AbstractGM
* `zmask`: specifies that values of z that are allowed to change during optimisation
* `u`: labels for optimisation
* `decoder`: decodes z into u-space
* `opt`: Flux optimiser
* `N`: optimisation steps
"""
# function reoptimize_latent!(model, u, z, opt, zmask; N=200)
function reidentify(z::AbstractMatrix, zmask::AbstractMatrix, u::AbstractMatrix,
                    decoder::AbstractCPDF, opt; N=200)
    zr = copy(z)
    ps = Flux.Params([zr])
    loss() = -sum(loglikelihood(decoder, u, zr))

    for ii in 1:N
        try
            d = Dict()
            gs = Flux.gradient(loss, ps)
            for p in ps
                g = gs[p]
                g[zmask] .= 0
                d[p] = g
            end
            gs = Zygote.Grads(d)
            Flux.Optimise.update!(opt, ps, gs)
            println("ii=$ii N=$N loss=$(loss())")
        catch ex
            rethrow(ex)
        end
    end
    return zr
end

# function reidentify(u, model, batch, opt, relevance_mask; N=100)
function reidentify(zmask::AbstractVector, u::AbstractVector, zbatchsize::Int,
                    model::AbstractGM, opt; N=100)
    U = repeat(u, inner=(1, zbatchsize))
    Z = rand(model.encoder, U)
    X = mean(model.decoder, Z)

    zlen = size(Z,1)
    Zmask = repeat(reshape(.!zmask, zlen, 1), inner=(1, zbatchsize))
    Zr = reidentify(Z, Zmask, U, model.decoder, opt, N=N)

    # get best result
    llh = dropdims(loglikelihood(model.decoder, U, Zr), dims=1)
    idx = argmax(llh)
    z   = Z[:,idx]
    x   = X[:,idx]
    zr  = Zr[:,idx]
    xr  = mean(model.decoder, zr)

    (z, x, zr, xr)
end

function relevance_mask(Z::AbstractMatrix, σmax=0.05, μmax=0.05)
    μ = [mean(Z[ii,:]) for ii in 1:size(Z,1)]
    σ = [std(Z[ii,:]) for ii in 1:size(Z,1)]
    s = σ .> σmax
    m = abs.(μ) .> μmax
    m .| s
end
