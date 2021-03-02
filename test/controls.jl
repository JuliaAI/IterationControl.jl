@testset "stopping criteria as controls" begin

    # A stopping criterion than ignores training losses:

    m = SquareRooter(4)
    c = NumberLimit(2)

    IC.train!(m, 3)
    state = IC.update!(c, m, 0)
    @test state == 1
    @test !IC.done(c, state)
    IC.train!(m, 3)
    state = IC.update!(c, m, 0, state)
    @test state == 2
    @test IC.done(c, state)
    report = @test_logs (:info, r"NumberLimit\(2\)") IC.takedown(c, 1, state)
    @test report.done
    @test report.log ==
        "Early stop triggered by NumberLimit(2) stopping criterion. "

    # A stopping criterion than uses training losses:

    m = SquareRooter(4)
    c = PQ()

    IC.train!(m, 3)
    state = IC.update!(c, m, 0)
    @test state.training_losses == reverse(m.training_losses)
    @test !IC.done(c, state)
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test state.training_losses == reverse(m.training_losses)
    @test !IC.done(c, state)
    report = IC.takedown(c, 1, state)
    @test !report.done
    @test report.log == ""

end

@testset "Train" begin
    m = SquareRooter(4)
    IC.train!(m, 10)
    all_training_losses = m.training_losses

    m = SquareRooter(4)
    c = Train(n=2)
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
    losses =

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
         log="Stopping early stop triggered by a `Callback` control. ")

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