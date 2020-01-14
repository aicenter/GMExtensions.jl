export latentboxplot
export annotatedheatmap
export plothistory
export plotreconstruction

@userplot LatentBoxplot
@recipe function f(b::LatentBoxplot)
    if length(b.args) != 2 ||
            !(typeof(b.args[1]) <: AbstractMatrix) ||
            !(typeof(b.args[2]) <: AbstractVector)
        error("latentboxplot needs a matrix Z and a vector of labels")
    end

    Z = b.args[1]
    labels = b.args[2]

    @series begin
        seriestype := :boxplot
        legend := false
        bar_width := 0.5
        whisker_width := 0.25

        repeat(labels, outer=size(Z,2)), reshape(Z,:)
    end
end

@userplot AnnotatedHeatmap
@recipe function f(h::AnnotatedHeatmap; annotationargs=(:white,))
    y = h.args[1]              #Get the input arguments, stored in h.args 
                               # - in this case there's only one
    typeof(y) <: AbstractMatrix || error("Pass a Matrix as the arg to heatmap")

    grid := false                      # turn off the background grid
    xaxis := false
    yaxis := false

    @series begin                      # the main series, showing the heatmap
        seriestype := :heatmap
        y
    end

    rows, cols = size(y)

    #horizontal lines
    for i in 0:cols         # each line is added as its own series, for clearer code
        @series begin
            seriestype := :path
            primary := false          # to avoid showing the lines in a legend
            linewidth := 2
            linecolor --> :white
            [i, i] .+ 0.5, [0, rows] .+ 0.5  # x and y values of lines
        end
    end

    for i in 0:rows
        @series begin
            seriestype := :path
            primary := false
            linewidth := 2
            linecolor --> :white
            [0, cols] .+ 0.5, [i,i] .+ 0.5
        end
    end

    @series begin
        seriestype := :scatter
        # make the points transparent - setting marker (or seriestype)
        # to :none doesn't currently work right
        markerstrokecolor := RGBA(0,0,0,0.)
        seriescolor := RGBA(0,0,0,0.)
        series_annotations := text.(round.(reshape(y,:), digits=2), annotationargs...)
        primary := false
        repeat(1:cols, inner = rows), repeat(1:rows, outer = cols)
    end
end

@userplot PlotHistory
@recipe function f(p::PlotHistory; exclude_keys=[:xrec])
    if !(typeof(p.args[1]) <: MVHistory)
        error("Can only plot MVHistories")
    end

    history = p.args[1]
    plot_keys = collect(keys(history))
    plot_keys = filter!(x -> !(x in exclude_keys), plot_keys)

    N = length(plot_keys)
    layout := grid(N, 1, heights=ones(N)./N)
    size := (500,1000)

    for (nr, key) in enumerate(plot_keys)
        x, y = get(history, key)
        if ndims(y[1]) == 2
            y = [z[:,1] for z in y]
        end
        y = hcat(y...)'

        @series begin
            seriestype := :path
            subplot := nr
            ylabel := key
            legend := false
            x, y
        end
    end
end


@userplot PlotReconstruction
@recipe function f(p::PlotReconstruction)
    if (length(p.args) != 2) ||
            !(typeof(p.args[1]) <: AbstractMatrix) ||
            !(typeof(p.args[2]) <: AbstractMatrix) ||
            (size(p.args[1]) != size(p.args[2]))
        error("Need two matrices of same size as arguments")
    end

    X = p.args[1]
    Xrec = p.args[2]
    @series begin
        linewidth := 2
        legend := false
        Xrec
    end

    @series begin
        color := "gray"
        linewidth := 2
        legend := false
        X
    end
end
