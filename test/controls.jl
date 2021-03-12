@testset "Step" begin
    m = SquareRooter(4)
    IC.train!(m, 10)
    all_training_losses = m.training_losses

    m = SquareRooter(4)
    c = Step(n=2)
    state = IC.update!(c, m, 0)
    @test state === nothing
    @test m.training_losses == all_training_losses[1:2]
    state = IC.update!(c, m, 0)
    @test m.training_losses == all_training_losses[3:4]
    @test !IC.done(c, state)
    @test IC.takedown(c, 1, state) == NamedTuple()
end

@testset "Info" begin
    m = SquareRooter(4)
    c = Info(m->m.root)
    IC.train!(m, 1)
    @test_logs (:info, 2.5) IC.update!(c, m, 1)
    @test_logs (:info, 2.5) IC.update!(c, m, 0)
    state = @test_logs IC.update!(c, m, -1)
    @test state === nothing
    @test_logs (:info, 2.5) IC.update!(c, m, 1, state)
    @test !IC.done(c, state)
    @test IC.takedown(c, 10, state) == NamedTuple()
end

@testset "Warn" begin
    m = SquareRooter(4)
    c = Warn(m -> m.root > 2.4)

    IC.train!(m, 1)
    @test_logs (:warn, "") IC.update!(c, m, 1)
    @test_logs (:warn, "") IC.update!(c, m, 0)
    state = @test_logs IC.update!(c, m, -1)
    @test state === ("", )

    IC.train!(m, 1)
    @test_logs  IC.update!(c, m, 1)
    @test_logs  IC.update!(c, m, 0)
    state = @test_logs IC.update!(c, m, -1)
    @test state === ()

    m = SquareRooter(4)
    IC.train!(m, 1)
    state =  IC.update!(c, m, -1)
    @test_logs (:warn, "") IC.update!(c, m, 1, state)
    @test_logs (:warn, "") IC.update!(c, m, 0, state)
    state = @test_logs IC.update!(c, m, -1, state)
    @test state === ("", "")

    IC.train!(m, 1)
    @test_logs  IC.update!(c, m, 1, state)
    @test_logs  IC.update!(c, m, 0, state)
    state = @test_logs IC.update!(c, m, -, state)
    @test state === ("", "")

    m = SquareRooter(4)
    c = Warn(m -> m.root > 2.4, f = m->m.root)

    IC.train!(m, 1)
    @test_logs (:warn, 2.5) IC.update!(c, m, 1)
    @test_logs (:warn, 2.5) IC.update!(c, m, 0)
    state = @test_logs IC.update!(c, m, -1)
    @test state === (2.5, )

    @test_logs (:warn, 2.5) IC.update!(c, m, 1, state)
    @test_logs (:warn, 2.5) IC.update!(c, m, 0, state)
    state = @test_logs IC.update!(c, m, -1, state)
    @test state === (2.5, 2.5)

    @test !IC.done(c, state)
    @test IC.takedown(c, 10, state) == (warnings = (2.5, 2.5),)
end

@testset "Error" begin
    m = SquareRooter(4)
    c = Error(m -> m.root > 2.4)

    IC.train!(m, 1)
    state = @test_logs (:error, "") IC.update!(c, m, 1)
    @test state === (done=true, error="")

    IC.train!(m, 1)
    state = @test_logs  IC.update!(c, m, 1)
    @test state === (done=false, error=())

    m = SquareRooter(4)
    IC.train!(m, 1)
    state = @test_logs (:error, "") IC.update!(c, m, 1)
    state = @test_logs (:error, "") IC.update!(c, m, 1, state)
    @test state === (done=true, error="")

    m = SquareRooter(4)
    c = Error(m -> m.root > 2.4, f = m->m.root)

    IC.train!(m, 1)
    state = @test_logs (:error, 2.5) IC.update!(c, m, 1)
    @test state === (done=true, error=2.5)

    IC.train!(m, 1)
    state = @test_logs  IC.update!(c, m, 1)
    @test state === (done=false, error=())

    m = SquareRooter(4)
    IC.train!(m, 1)
    state = @test_logs (:error, 2.5) IC.update!(c, m, 1)
    state = @test_logs (:error, 2.5) IC.update!(c, m, 1, state)
    @test state === (done=true, error=2.5)

    @test IC.done(c, state)
    @test IC.takedown(c, 10, state) == state
end

@testset "Callback" begin

    v = Float64[]
    f(model) = (push!(v, IC.loss(model)); last(v) < 0.02)

    c = Callback(f)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test !state.done
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 0, state) == (done = false, log="")

    v = Float64[]
    f(model) = (push!(v, IC.loss(model)); last(v) < 0.02)

    c = Callback(f, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test state.done
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="Stop triggered by a `Callback` control. ")

    v = Float64[]
    f(model) = (push!(v, IC.loss(model)); last(v) < 0.02)

    c = Callback(f, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test state.done
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="foo")

end

@testset "WithLossDo" begin

    v = Float64[]
    f(loss) = (push!(v, loss); last(v) < 0.02)
    c = WithLossDo(f)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test !state.done
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 0, state) == (done = false, log="")

    v = Float64[]
    f(loss) = (push!(v, loss); last(v) < 0.02)
    c = WithLossDo(f, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test state.done
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="Stop triggered by a `WithLossDo` control. ")

    v = Float64[]
    f(loss) = (push!(v, loss); last(v) < 0.02)
    c = WithLossDo(f, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test state.done
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="foo")

end

@testset "WithTrainingLossesDo" begin

    v = Float64[]
    f(training_loss) = (push!(v, last(training_loss)); last(v) < 0.5)
    c = WithTrainingLossesDo(f)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v ≈ [1.5, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 0, state)
    @test !state.done
    @test v ≈ [1.5, 0.45]
    @test IC.takedown(c, 0, state) == (done = false, log="")

    v = Float64[]
    f(training_loss) = (push!(v, last(training_loss)); last(v) < 0.5)
    c = WithTrainingLossesDo(f, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [1.5, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 0, state)
    @test state.done
    @test v ≈ [1.5, 0.45]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="Stop triggered by a `WithTrainingLossesDo` control. ")

    v = Float64[]
    f(training_loss) = (push!(v, last(training_loss)); last(v) < 0.5)
    c = WithTrainingLossesDo(f, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [1.5, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 0, state)
    @test state.done
    @test v ≈ [1.5, 0.45]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="foo")
end


@testset "WithNumberDo" begin
    v = Int[]
    f(n) = (push!(v, n); last(n) > 1)
    c = WithNumberDo(f)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [1, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 0, state)
    @test !state.done
    @test v == [1, 2]
    @test IC.takedown(c, 0, state) == (done = false, log="")

    v = Int[]
    f(n) = (push!(v, n); last(n) > 1)
    c = WithNumberDo(f, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [1, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 0, state)
    @test state.done
    @test v == [1, 2]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="Stop triggered by a `WithNumberDo` control. ")

    v = Int[]
    f(n) = (push!(v, n); last(n) > 1)
    c = WithNumberDo(f, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [1, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 0, state)
    @test state.done
    @test v == [1, 2]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="foo")
end

@testset "Data" begin
    data = Float64[1.0, -0.9, 0]

    for option in [true, false]
        model = Particle(0.1)
        c = Data(data, stop_when_exhausted=option)

        state = IC.update!(c, model, 0)
        IC.train!(model, 1)
        @test loss(model) ≈ 0.9

        state = IC.update!(c, model, 0, state)
        IC.train!(model, 1)
        @test loss(model) ≈ 0.9

        state = IC.update!(c, model, 0, state)
        IC.train!(model, 1)
        @test loss(model) ≈ 0.0

        @test !IC.done(c, state)

        if option
            state = IC.update!(c, model, 0, state)
            @test IC.done(c, state)
            report = @test_logs (:info, IC.DATA_STOP) IC.takedown(c, 1, state)
            @test report == (done = true, log = IC.DATA_STOP)
        else
            state = @test_logs IC.update!(c, model, 0, state)
            @test !IC.done(c, state)
            report = IC.takedown(c, 1, state)
            @test report == (done = false, log = "")
        end

    end
end

@testset "integration test" begin
    data = repeat([-1, 1], outer=5);

    model = Particle(0.1)
    losses = Float64[]
    callback!(model) = push!(losses, model.position)
    report = IC.train!(model,
                       Data(data),
                       Step(5),
                       Threshold(0.01),
                       TimeLimit(0.0005),
                       Info(loss),
                       Callback(callback!);
                       verbosity=-1)
    @test !report[1][2].done
    @test report[3][2].done
    @test loss(model) < 0.01

    model = Particle(0.1)
    losses = Float64[]
    report = IC.train!(model,
                       Data(data, stop_when_exhausted=true),
                       Step(5),
                       Threshold(0.01),
                       TimeLimit(0.0005),
                       Info(loss),
                       Callback(callback!),
                       verbosity=-1)
    @test length(losses) == length(data) + 1
    @test loss(model) > 0.01
end
