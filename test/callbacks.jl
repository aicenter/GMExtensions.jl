@testset "callbacks" begin
    unicodeplots()

    enc = CMeanGaussian{DiagVar}(Dense(5,2), ones(Float32,2))
    dec = CMeanVarGaussian{ScalarVar}(Dense(2,6))
    model = VAE(2, enc, dec)
    loss(x) = elbo(model, x)
    opt = ADAM()

    # setup progress printing
    niter = 10
    test_data = rand(Float32, 5, 10)
    prog = Progress(niter)
    curr_iter = zeros(Int)

    callbacks = [
      GMExtensions.progress_callback(prog, curr_iter, model, test_data, loss),
      () -> (sleep(0.2))
    ]

    for ii in 1:niter
        curr_iter .+= 1
        test_data .= rand(Float32, 5, 10)
        for cb in callbacks cb() end
    end

    @test curr_iter[1] == niter
end
