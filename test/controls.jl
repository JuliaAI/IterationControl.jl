@testset "Step" begin
    m = SquareRooter(4)
    IC.train!(m, 10)
    all_training_losses = m.training_losses

    m = SquareRooter(4)
    c = Step(n=2)
    state = IC.update!(c, m, 0, 1)
    @test state === (new_iterations = 2,)
    @test m.training_losses == all_training_losses[1:2]
    state = IC.update!(c, m, 0, 2, state)
    @test m.training_losses == all_training_losses[3:4]
    @test !IC.done(c, state)
    @test_logs((:info, r"A total of "),
               @test IC.takedown(c, 2, state) == (new_iterations = 4,))
end

@testset "Info" begin
    m = SquareRooter(4)
    c = Info(m->m.root)
    IC.train!(m, 1)
    @test_logs (:info, 2.5) IC.update!(c, m, 2, 1)
    @test_logs (:info, 2.5) IC.update!(c, m, 1, 1)
    state = @test_logs IC.update!(c, m, 0, 1)
    @test state === nothing
    @test_logs (:info, 2.5) IC.update!(c, m, 2, 2, state)
    @test !IC.done(c, state)
    @test IC.takedown(c, 10, state) == NamedTuple()
end

@testset "Warn" begin
    m = SquareRooter(4)
    c = Warn(m -> m.root > 2.4)

    IC.train!(m, 1)
    @test_logs (:warn, "") IC.update!(c, m, 1, 1)
    @test_logs (:warn, "") IC.update!(c, m, 0, 1)
    state = @test_logs IC.update!(c, m, -1, 1)
    @test state === (warnings=("", ),)

    IC.train!(m, 1)
    @test_logs  IC.update!(c, m, 1, 1)
    @test_logs  IC.update!(c, m, 0, 1)
    state = @test_logs IC.update!(c, m, -1, 1)
    @test state === (warnings=(),)

    m = SquareRooter(4)
    IC.train!(m, 1)
    state =  IC.update!(c, m, -1, 1)
    @test_logs (:warn, "") IC.update!(c, m, 1, 2, state)
    @test_logs (:warn, "") IC.update!(c, m, 0, 2, state)
    state = @test_logs IC.update!(c, m, -1, 2, state)
    @test state === (warnings=("", ""),)

    IC.train!(m, 1)
    @test_logs  IC.update!(c, m, 1, 3, state)
    @test_logs  IC.update!(c, m, 0, 3, state)
    state = @test_logs IC.update!(c, m, -1, 3, state)
    @test state === (warnings=("", ""),)

    m = SquareRooter(4)
    c = Warn(m -> m.root > 2.4, f = m->m.root)

    IC.train!(m, 1)
    @test_logs (:warn, 2.5) IC.update!(c, m, 1, 1)
    @test_logs (:warn, 2.5) IC.update!(c, m, 0, 1)
    state = @test_logs IC.update!(c, m, -1, 1)
    @test state === (warnings=(2.5, ),)

    @test_logs (:warn, 2.5) IC.update!(c, m, 1, 2, state)
    @test_logs (:warn, 2.5) IC.update!(c, m, 0, 2, state)
    state = @test_logs IC.update!(c, m, -1, 2, state)
    @test state === (warnings=(2.5, 2.5),)

    @test !IC.done(c, state)
    @test_logs((:warn, r"A `Warn`"),
               @test IC.takedown(c, 2, state) == (warnings = (2.5, 2.5),))
    @test_logs @test IC.takedown(c, 1, state) == (warnings = (2.5, 2.5),)

end

@testset "Error" begin
    m = SquareRooter(4)
    c = Error(m -> m.root > 2.4)

    IC.train!(m, 1)
    state = @test_logs (:error, "") IC.update!(c, m, 2, 1)
    @test state === (done=true, error="")

    IC.train!(m, 1)
    state = @test_logs  IC.update!(c, m, 2, 1)
    @test state === (done=false, error=())

    m = SquareRooter(4)
    IC.train!(m, 1)
    state = @test_logs (:error, "") IC.update!(c, m, 2, 1)
    state = @test_logs (:error, "") IC.update!(c, m, 2, 2, state)
    @test state === (done=true, error="")

    m = SquareRooter(4)
    c = Error(m -> m.root > 2.4, f = m->m.root)

    IC.train!(m, 1)
    state = @test_logs (:error, 2.5) IC.update!(c, m, 2, 1)
    @test state === (done=true, error=2.5)

    IC.train!(m, 1)
    state = @test_logs  IC.update!(c, m, 2, 1)
    @test state === (done=false, error=())

    m = SquareRooter(4)
    IC.train!(m, 1)
    state = @test_logs (:error, 2.5) IC.update!(c, m, 2, 1)
    state = @test_logs (:error, 2.5) IC.update!(c, m, 2, 2, state)
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
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
    @test !state.done
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 0, state) == (done = false, log="")

    v = Float64[]
    f2(model) = (push!(v, IC.loss(model)); last(v) < 0.02)

    c = Callback(f2, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
    @test state.done
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="Stop triggered by a `Callback` control. ")

    v = Float64[]
    f3(model) = (push!(v, IC.loss(model)); last(v) < 0.02)

    c = Callback(f3, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
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
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
    @test !state.done
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 1, state) ==
        (loss=v[end], done = false, log="")
    @test_logs((:info, r"final loss"),
               @test IC.takedown(c, 2, state) ==
               (loss=v[end], done = false, log=""))

    v = Float64[]
    f2(loss) = (push!(v, loss); last(v) < 0.02)
    c = WithLossDo(f2, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
    @test state.done
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 0, state) ==
        (loss = v[end],
         done = true,
         log="Stop triggered by a `WithLossDo` control. ")
    @test_logs((:info, r"Stop triggered"),
               @test IC.takedown(c, 1, state) ==
               (loss = v[end],
                done = true,
                log="Stop triggered by a `WithLossDo` control. "))
    @test_logs((:info, r"final loss"),
               (:info, r"Stop triggered"),
               @test IC.takedown(c, 2, state) ==
               (loss = v[end],
                done = true,
                log="Stop triggered by a `WithLossDo` control. "))

    v = Float64[]
    f3(loss) = (push!(v, loss); last(v) < 0.02)
    c = WithLossDo(f3, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
    @test state.done
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 0, state) ==
        (loss = v[end],
         done = true,
         log="foo")

end

@testset "WithTrainingLossesDo" begin

    v = Float64[]
    f(training_loss) = (push!(v, last(training_loss)); last(v) < 0.5)
    c = WithTrainingLossesDo(f)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v ≈ [1.5, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 2, state)
    @test !state.done
    @test v ≈ [1.5, 0.45]
    @test IC.takedown(c, 0, state) ==
        (latest_training_loss = v[end], done = false, log="")
    @test_logs((:info, r"final train"),
               @test IC.takedown(c, 2, state) ==
               (latest_training_loss=v[end], done = false, log=""))


    v = Float64[]
    f1(training_loss) = (push!(v, last(training_loss)); last(v) < 0.5)
    c = WithTrainingLossesDo(f1, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v == [1.5, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 2, state)
    @test state.done
    @test v ≈ [1.5, 0.45]
    @test IC.takedown(c, 0, state) ==
        (latest_training_loss = v[end],
         done = true,
         log="Stop triggered by a `WithTrainingLossesDo` control. ")
    @test_logs((:info, r"Stop "),
               @test IC.takedown(c, 1, state) ==
               (latest_training_loss = v[end],
                done = true,
                log="Stop triggered by a `WithTrainingLossesDo` control. "))
    @test_logs((:info, r"final train"),
               (:info, r"Stop"),
               @test IC.takedown(c, 2, state) ==
               (latest_training_loss = v[end],
                done = true,
                log="Stop triggered by a `WithTrainingLossesDo` control. "))

    v = Float64[]
    f2(training_loss) = (push!(v, last(training_loss)); last(v) < 0.5)
    c = WithTrainingLossesDo(f2, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v == [1.5, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 2, state)
    @test state.done
    @test v ≈ [1.5, 0.45]
    @test IC.takedown(c, 0, state) ==
        (latest_training_loss = v[end],
         done = true,
         log="foo")
end


@testset "WithNumberDo" begin
    v = Int[]
    f(n) = (push!(v, n); last(n) > 1)
    c = WithNumberDo(f)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v == [1, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 2, state)
    @test !state.done
    @test v == [1, 2]
    @test IC.takedown(c, 0, state) == (done = false, n = 2, log="")
    @test_logs((:info, r"final number"),
               @test IC.takedown(c, 2, state) == (done = false, n = 2, log=""))

    v = Int[]
    f2(n) = (push!(v, n); last(n) > 1)
    c = WithNumberDo(f2, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v == [1, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 2, state)
    @test state.done
    @test v == [1, 2]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         n= 2,
         log="Stop triggered by a `WithNumberDo` control. ")
    @test_logs((:info, r"Stop"),
               @test IC.takedown(c, 1, state) ==
               (done = true,
                n= 2,
                log="Stop triggered by a `WithNumberDo` control. "))
    @test_logs((:info, r"final number"),
               (:info, r"Stop"),
               @test IC.takedown(c, 2, state) ==
               (done = true,
                n= 2,
                log="Stop triggered by a `WithNumberDo` control. "))

    v = Int[]
    f3(n) = (push!(v, n); last(n) > 1)
    c = WithNumberDo(f3, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !state.done
    @test v == [1, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 2, state)
    @test state.done
    @test v == [1, 2]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         n = 2,
         log="foo")
end

@testset "Data" begin
    data = Float64[1.0, -0.9, 0]

    for option in [true, false]
        model = Particle(0.1)
        c = Data(data, stop_when_exhausted=option)

        state = IC.update!(c, model, 0, 1)
        IC.train!(model, 1)
        @test loss(model) ≈ 0.9

        state = IC.update!(c, model, 0, 2, state)
        IC.train!(model, 1)
        @test loss(model) ≈ 0.9

        state = IC.update!(c, model, 0, 3, state)
        IC.train!(model, 1)
        @test loss(model) ≈ 0.0

        @test !IC.done(c, state)

        if option
            state = IC.update!(c, model, 0, 4, state)
            @test IC.done(c, state)
            report = @test_logs (:info, IC.DATA_STOP) IC.takedown(c, 1, state)
            @test report == (done = true, log = IC.DATA_STOP)
        else
            state = @test_logs IC.update!(c, model, 0, 4, state)
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
    noise = fill((:info, r""), 33)
    report = @test_logs(noise...,
                        IC.train!(model,
                                  Data(data, stop_when_exhausted=true),
                                  Step(5),
                                  WithNumberDo(),
                                  WithLossDo(),
                                  WithTrainingLossesDo(),
                                  Threshold(0.01),
                                  TimeLimit(0.0005),
                                  Info(loss),
                                  Callback(callback!),
                                  verbosity=-1))
    @test length(losses) == length(data) + 1
    @test loss(model) > 0.01
end

@testset "constructors #55" begin
    for Control in [WithTrainingLossesDo,
                    Callback,
                    WithLossDo,
                    WithNumberDo]
        g(x) = true

        c = Control(g)
        @test c.f == g
        @test !c.stop_if_true

        c= Control(g, stop_if_true=true)
        @test c.f == g
        @test c.stop_if_true

        c= Control(f=g)
        @test c.f == g
        @test !c.stop_if_true

        c= Control(f=g, stop_if_true=true)
        @test c.f == g
        @test c.stop_if_true
    end
end
