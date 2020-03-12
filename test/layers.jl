@testset "LinearMap" begin
    m = LinearMap(3,2)
    ps = params(m)
    @test length(ps) == 1
    @test size(m(rand(3,10))) == (2,10)
end

@testset "SplitLayer" begin
    m = SplitLayer(2, [3,4])
    @test length(params(m)) == 4

    (a,b) = m(rand(2,10))
    @test size(a) == (3,10)
    @test size(b) == (4,10)
end

@testset "CatLayer" begin
    m = CatLayer(Dense(3,2), Dense(3,4))
    @test length(params(m)) == 4
    @test size(m(rand(3,10))) == (6,10)
end

@testset "Bias" begin
    m = Bias(3)
    @test length(params(m)) == 1
    @test size(m(rand(3))) == (3,)
end
