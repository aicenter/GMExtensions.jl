@testset "utils/saveload.jl" begin

    xlen = 3
    zlen = 2
    keep = 2

    enc = ConditionalMvNormal(Dense(xlen, zlen))
    dec = ConditionalMvNormal(Dense(zlen, xlen))
    model = VAE(zlen, enc, dec)

    nowarn_logger = SimpleLogger(stdout, Logging.Error)
    model_dir   = mktempdir()
    @debug "  model_dir: $model_dir"


    @debug "  Testing `save_checkpoint`"
    model_ckpt = joinpath(model_dir, "ckpt.bson")
    history = MVHistory()
    push!(history, :loss, 1, 1)
    with_logger(nowarn_logger) do
        save_checkpoint(model_ckpt, model, history, keep=keep)
        save_checkpoint(model_ckpt, model, history, keep=keep)
        save_checkpoint(model_ckpt, model, history, keep=keep)
    end

    @test isfile(model_ckpt)
    @test length(readdir(dirname(model_ckpt))) == keep


    @debug "  Testing `load_checkpoint`"
    loaded_model, history = with_logger(nowarn_logger) do 
        load_checkpoint(model_ckpt)
    end
    @test model.encoder.mapping.W == loaded_model.encoder.mapping.W

    opt = ADAM()
    lossf(x) = -elbo(model, x, β=1e-3)
    data = [(randn(Float32, xlen, 10),)]
    Flux.train!(lossf, params(model), data, opt)
    params_trained = get_params(model)

    with_logger(nowarn_logger) do
        save_checkpoint(model_ckpt, model, history, keep=keep)
    end

    loaded_model, history = with_logger(nowarn_logger) do 
        load_checkpoint(model_ckpt)
    end

    @test length(params(model)) == length(params(loaded_model))
    @test !any(param_change(params_trained, loaded_model)) # did the params change?
    @test size(mean(loaded_model.encoder, randn(Float32, xlen))) == (zlen,)
end
