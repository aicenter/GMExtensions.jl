export progress_callback
export mvhistory_callback

"""
    progress_callback(prog::Progress, iter::Array{Int,0}, model::AbstractGM,
                           test_data::AbstractMatrix, loss::Function)

Prints progressbar, latent distribution, and reconstructions.
Preferably used with UnicodePlots.
"""
function progress_callback(prog::Progress, iter::Array{Int,0}, model::AbstractGM,
                           test_data::AbstractMatrix, loss::Function)
    function callback()
        vmin = minimum(test_data)
        vmax = maximum(test_data)
        μz = mean(model.encoder, test_data);
        p1 = latentboxplot(μz, 1:size(μz,1), size=(700,300), ylimits=(-2,2));
        p2 = plotreconstruction(test_data[:,1:3], mean(model.decoder, μz[:,1:3]),
                                 size=(700,300), ylimits=(vmin,vmax));
        plt = plot(p1, p2, layout=(2,1))
        sv = [(:iter, iter),
              (:loss, loss(test_data)),
              (:plot, repr(display(plt)))]
        ProgressMeter.update!(prog, iter[1]; showvalues=sv)
    end
end

function push_ntuple!(history::MVHistory, ntuple::NamedTuple; idx=nothing)
    if idx == nothing
        _keys = keys(history)
        if length(_keys) > 0
            idx = length(history, first(_keys)) + 1
        else
            idx = 1
        end
    end

    for (name, value) in pairs(ntuple)
        push!(history, name, idx, deepcopy(value))
    end
end

function mvhistory_callback(h::MVHistory,
                            m::AbstractVAE,
                            lossf::Function, test_data::AbstractArray)
    function callback()
        (μz, σz) = mean_var(m.encoder, test_data)
        λz = variance(m.prior)
        μx, σx = mean_var(m.decoder, μz)
        loss = lossf(test_data)

        μz = μz |> cpu
        σz = σz |> cpu
        λz = λz |> cpu
        μx = μx |> cpu
        loss = loss |> cpu
        σx  = σx |> cpu

        ntuple = @ntuple μz σz λz μx loss σx test_data
        push_ntuple!(h, ntuple)
    end
end
