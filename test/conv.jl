@testset "convolutional constructors" begin
    s = (8, 6, 2, 6)
    x = randn(Float32, s) |> gpu
    xsize = s[1:3]
    ldim = 2
    kernelsizes = (3, 5)
    nchannels = (4, 8)
    scalings = (2, 1)
    densedims = [128]
    
    # encoder
    encoder = GMExtensions.conv_encoder(xsize, ldim, kernelsizes, nchannels, scalings) |> gpu
    y = encoder(x)
    @test length(encoder.layers) == 2*length(kernelsizes) + 1 + 1 # 2*(conv + maxpool) + reshape + dense
    @test size(y) == (ldim, s[end])
    @test eltype(y) == Float32

    encoder = GMExtensions.conv_encoder(xsize, ldim, kernelsizes, nchannels, scalings; 
        densedims = densedims) |> gpu
    @test length(encoder.layers) == 2*length(kernelsizes) + 1 + 2 # 2*(conv + maxpool) + reshape + 2*dense
    
    # decoder
    decoder = GMExtensions.conv_decoder(xsize, ldim, reverse(kernelsizes), 
        reverse(nchannels), reverse(scalings)) |> gpu
    z = decoder(y)
    @test length(decoder.layers) == 1 + 1 + 2*length(kernelsizes) # dense + reshape + 2*(conv + maxpool)
    @test size(z) == size(x)
    @test eltype(z) == Float32

    decoder = GMExtensions.conv_decoder(xsize, ldim, reverse(kernelsizes), 
        reverse(nchannels), reverse(scalings); densedims = densedims) |> gpu
    @test length(decoder.layers) == 2 + 1 + 2*length(kernelsizes) # dense + reshape + 2*(conv + maxpool)
    
    # also test trainability
    enc_params = get_params(encoder)
    dec_params = get_params(decoder)    
    loss(x) = Flux.mse(decoder(encoder(x)), x)
    opt = ADAM()
    data = (x,)
    GenerativeModels.update_params!(encoder, data, loss, opt)
    @test all(param_change(enc_params, encoder))
    GenerativeModels.update_params!(decoder, data, loss, opt)
    @test all(param_change(dec_params, decoder))

    # test vectorization of output
    decoder = GMExtensions.conv_decoder(xsize, ldim, reverse(kernelsizes), 
        reverse(nchannels), reverse(scalings); densedims = densedims, vec_output = true) |> gpu
    z = decoder(y)
    @test size(z) == (reduce(*, xsize), 6)
    @test size(reshape(z, s...)) == s

    decoder = GMExtensions.conv_decoder(xsize, ldim, reverse(kernelsizes), 
        reverse(nchannels), reverse(scalings); densedims = densedims, vec_output = true,
        vec_output_dim = reduce(*, xsize) + 1) |> gpu
    z = decoder(y)
    @test size(z) == (reduce(*, xsize) + 1, 6)

end 

