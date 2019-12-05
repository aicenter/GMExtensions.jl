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
        plt = plot(p1, p2)
        sv = [(:iter, iter),
              (:loss, loss(test_data)),
              (:plot, repr(display(plt)))]
        ProgressMeter.update!(prog, iter[1]; showvalues=sv)
    end
end
