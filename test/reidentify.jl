@testset "reidentify" begin
    enc = CMeanVarGaussian{Float32,ScalarVar}(Dense(4,5))
    dec = CMeanVarGaussian{Float32,ScalarVar}(Dense(4,5))
    model = VAE(4, enc, dec)
    opt = ADAM()

    N = 10
    zmask = reshape([1 0 0 0] .== 1, :)
    u = rand(Float32, 4)

    (z, x, zr, xr) = GMExtensions.reidentify(zmask, u, 2, model, opt, N=N)

    @test all(z[zmask] .!= zr[zmask])
    @test all(z[.!zmask] .== zr[.!zmask])
end
