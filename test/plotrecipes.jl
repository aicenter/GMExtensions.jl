@testset "plot recipes" begin
    gr()

    Z = randn(5, 100)
    labels = ["a", "asdf", "b", "c", "d"]
    latentboxplot(Z,labels)
    
    annotatedheatmap(rand(3,3),)
     
    h = MVHistory()
    for _ in 1:100
        push!(h, :a, rand(3))
        push!(h, :b, rand(3))
        push!(h, :c, rand(3))
        push!(h, :c, rand(3))
        push!(h, :xrec, ones(3))
    end
    
    plothistory(h)

    plotreconstruction(rand(100,2), rand(100,2))
    @test true
end
