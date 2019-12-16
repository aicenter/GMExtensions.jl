@testset "convolutional constructors" begin
    s = (8, 6, 2, 6)
    x = randn(Float32, s)
    xsize = s[1:3]
    ldim = 2
    kernelsizes = (3, 5)
    nchannels = (4, 8)
    scalings = (2, 1)
    densedims = [128]
    
    # encoder
    encoder = GMExtensions.conv_encoder(xsize, ldim, kernelsizes, nchannels, scalings)
    y = encoder(x)
    @test length(encoder.layers) == 2*length(kernelsizes) + 1 + 1 # 2*(conv + maxpool) + reshape + dense
    @test size(y) == (ldim, s[end])
    @test eltype(y) == Float32

    encoder = GMExtensions.conv_encoder(xsize, ldim, kernelsizes, nchannels, scalings; 
        densedims = densedims)
    @test length(encoder.layers) == 2*length(kernelsizes) + 1 + 2 # 2*(conv + maxpool) + reshape + 2*dense
    
    # decoder
    decoder = GMExtensions.conv_decoder(xsize, ldim, reverse(kernelsizes), 
        reverse(nchannels), reverse(scalings))
    z = decoder(y)
    @test length(decoder.layers) == 1 + 1 + 2*length(kernelsizes) # dense + reshape + 2*(conv + maxpool)
    @test size(z) == size(x)
    @test eltype(z) == Float32

    decoder = GMExtensions.conv_decoder(xsize, ldim, reverse(kernelsizes), 
        reverse(nchannels), reverse(scalings); densedims = densedims)
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
end 

