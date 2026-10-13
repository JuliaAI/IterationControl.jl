using Dates
using IterationControl
import IterationControl as IC
import IterationControl.SquareRooter

@testset "Step" begin
    m = SquareRooter(4)
    IC.train!(m, 10)
    all_training_losses = m.training_losses

    m = SquareRooter(4)
    c = Step(n=2)
    state = IC.update!(c, m, 0, 1)
    @test state === (new_iterations = 2,)
    @test IC.public_state(c, state) == (; new_iterations=2, new_steps=1)
    @test m.training_losses == all_training_losses[1:2]
    state = IC.update!(c, m, 0, 2, state)
    @test IC.public_state(c, state) == (; new_iterations=4, new_steps=2)
    @test m.training_losses == all_training_losses[3:4]
    @test !IC.done(c, state)
    @test IC.takedown(c, 1, state) == ""
end

@testset "Info" begin
    m = SquareRooter(4)
    c = Info(m->m.root)
    IC.train!(m, 1)
    @test_logs (:info, 2.5) IC.update!(c, m, 2, 1)
    @test_logs (:info, 2.5) IC.update!(c, m, 1, 1)
    state = @test_logs IC.update!(c, m, 0, 1)
    @test state === 2.5
    @test IC.public_state(c, state) == (; called_back=2.5)
    @test_logs (:info, 2.5) IC.update!(c, m, 2, 2, state)
    @test !IC.done(c, state)
    @test IC.takedown(c, 10, state) == ""
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
    @test IC.public_state(c, state) == (;)

    m = SquareRooter(4)
    IC.train!(m, 1)
    state =  IC.update!(c, m, -1, 1)
    @test_logs (:warn, "") IC.update!(c, m, 1, 2, state)
    @test_logs (:warn, "") IC.update!(c, m, 0, 2, state)
    state = @test_logs IC.update!(c, m, -1, 2, state)
    @test state === (warnings=("", ""),)
    @test IC.public_state(c, state) == (;)

    IC.train!(m, 1)
    @test_logs  IC.update!(c, m, 1, 3, state)
    @test_logs  IC.update!(c, m, 0, 3, state)
    state = @test_logs IC.update!(c, m, -1, 3, state)
    @test state === (warnings=("", ""),)
    @test IC.public_state(c, state) == (;)

    m = SquareRooter(4)
    c = Warn(m -> m.root > 2.4, f = m->m.root)

    IC.train!(m, 1)
    @test_logs (:warn, 2.5) IC.update!(c, m, 1, 1)
    @test_logs (:warn, 2.5) IC.update!(c, m, 0, 1)
    state = @test_logs IC.update!(c, m, -1, 1)
    @test state === (warnings=(2.5, ),)
    @test IC.public_state(c, state) == (;)

    @test_logs (:warn, 2.5) IC.update!(c, m, 1, 2, state)
    @test_logs (:warn, 2.5) IC.update!(c, m, 0, 2, state)
    state = @test_logs IC.update!(c, m, -1, 2, state)
    @test state === (warnings=(2.5, 2.5),)
    @test IC.public_state(c, state) == (;)

    @test !IC.done(c, state)
    @test_logs(
        (:warn, r"A `Warn`"),
        (@test IC.takedown(c, 1, state) == "A `Warn` control issued 2 warnings."),
    )
end

@testset "Error" begin
    m = SquareRooter(4)
    c = Error(m -> m.root > 2.4)

    # error encountered after first train:
    IC.train!(m, 1)
    state = @test_logs (:error, "") IC.update!(c, m, 0, 1)
    @test state === (done=true, error="")
    @test IC.public_state(c, state) == (; error="")
    @test IC.done(c, state)


    # start again, with a model that won't throw errors after training:
    IC.train!(m, 1)
    state = @test_logs  IC.update!(c, m, 2, 1)
    @test state === (done=false, error=nothing)
    @test IC.public_state(c, state) == (; error=nothing)

    m = SquareRooter(4)
    IC.train!(m, 1)
    state = @test_logs (:error, "") IC.update!(c, m, 0, 1)
    state = @test_logs (:error, "") IC.update!(c, m, 0, 2, state)
    @test state === (done=true, error="")
    @test IC.public_state(c, state) == (; error="")

    m = SquareRooter(4)
    c = Error(m -> m.root > 2.4, f = m->m.root)

    IC.train!(m, 1)
    state = @test_logs (:error, 2.5) IC.update!(c, m, 0, 1)
    @test state === (done=true, error=2.5)
    @test IC.public_state(c, state) == (; error=2.5)

    IC.train!(m, 1)
    state = @test_logs  IC.update!(c, m, 0, 1)
    @test state === (done=false, error=nothing)
    @test IC.public_state(c, state) == (; error=nothing)

    m = SquareRooter(4)
    IC.train!(m, 1)
    state = @test_logs (:error, 2.5) IC.update!(c, m, 0, 1)
    state = @test_logs (:error, 2.5) IC.update!(c, m, 0, 2, state)
    @test state === (done=true, error=2.5)

    @test IC.done(c, state)
    @test IC.takedown(c, 10, state) == "2.5"
end

@testset "Callback" begin
    v = Float64[]
    function f(model)
        push!(v, IC.loss(model))
        return last(v) < 0.02
    end

    c = Callback(f)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test IC.public_state(c, state) == (; called_back=false)
    @test !IC.done(c, state)
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
    @test IC.public_state(c, state) == (; called_back=true)
    @test !IC.done(c, state) # because `c.stop_if_true==false`
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 0, state) == ""

    v = Float64[]
    c = Callback(f, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !IC.done(c, state)
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
    @test IC.done(c, state)
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test_logs(
        (:info, r"^Stop triggered by a `Callback"),
        (@test IC.takedown(c, 1, state) ==
            "Stop triggered by a `Callback` control. "),
    )

    v = Float64[]
    c = Callback(f, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !IC.done(c, state)
    @test IC.public_state(c, state) == (; called_back=false)
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
    @test IC.done(c, state)
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test_logs(
        (:info, "foo"),
        (@test IC.takedown(c, 1, state) == "foo"),
    )
end

@testset "WithLossDo" begin
    v = Float64[]
    f(loss) = (push!(v, loss); last(v) < 0.02)
    c = WithLossDo(f)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test IC.public_state(c, state) ==
        (loss=2.25, called_back=false)
    @test !IC.done(c, state)
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
    @test !IC.done(c, state) # because `stop_if_true=false`
    @test IC.public_state(c, state)[2]
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test IC.takedown(c, 1, state) == ""

    v = Float64[]
    c = WithLossDo(f, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !IC.done(c, state)
    @test v == [2.25, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
    @test IC.done(c, state)
    @test v ≈ [2.25, (3281/1640)^2 - 4]
    @test_logs(
        (:info, r"^Stop triggered by a `WithLossDo"),
        (@test IC.takedown(c, 1, state) ==
        "Stop triggered by a `WithLossDo` control. "),
    )

    v = Float64[]
    c = WithLossDo(f, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 3)
    state = IC.update!(c, m, 1, 2, state)
    @test IC.done(c, state)
    @test_logs(
        (:info, r"foo"),
        (@test IC.takedown(c, 1, state) == "foo"),
    )
end

@testset "WithTrainingLossesDo" begin
    v = Float64[]
    f(training_loss) = (push!(v, last(training_loss)); last(v) < 0.5)
    c = WithTrainingLossesDo(f)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test IC.public_state(c, state) ==
        (training_loss=1.5, called_back=false)
    @test !IC.done(c, state)
    @test v == [1.5, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 2, state)
    @test !IC.done(c, state) # because `stop_if_true=false`
    @test IC.public_state(c, state)[2]
    @test v ≈ [1.5, 0.45]
    @test IC.takedown(c, 1, state) == ""

    v = Float64[]
    c = WithTrainingLossesDo(f, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !IC.done(c, state)
    @test v == [1.5, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 2, state)
    @test IC.done(c, state)
    @test v ≈ [1.5, 0.45]
    @test_logs(
        (:info, r"^Stop triggered by a `WithTrainingLossesDo"),
        (@test IC.takedown(c, 1, state) ==
        "Stop triggered by a `WithTrainingLossesDo` control. "),
    )

    v = Float64[]
    c = WithTrainingLossesDo(f, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 2)
    state = IC.update!(c, m, 1, 2, state)
    @test IC.done(c, state)
    @test_logs(
        (:info, r"foo"),
        (@test IC.takedown(c, 1, state) == "foo"),
    )
end

@testset "WithNumberDo" begin
    v = Int[]
    f(n) = (push!(v, n); last(n) > 1)
    c = WithNumberDo(f)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test IC.public_state(c, state) == (; number=1, called_back=false)
    @test !IC.done(c, state)
    @test v == [1, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 2, state)
    @test IC.public_state(c, state) == (; number=2, called_back=true)
    @test !IC.done(c, state) # because `stop_if_true` is `false`
    @test v == [1, 2]
    @test_logs @test IC.takedown(c, 0, state) == ""

    v = Int[]
    c = WithNumberDo(f, stop_if_true=true)
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !IC.done(c, state)
    @test v == [1, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 2, state)
    @test IC.done(c, state)
    @test IC.public_state(c, state) == (; number=2, called_back=true)
    @test v == [1, 2]
    @test_logs(
        (:info, r"^Stop"),
        (@test IC.takedown(c, 1, state) ==
            "Stop triggered by a `WithNumberDo` control. "),
    )

    v = Int[]
    c = WithNumberDo(f, stop_if_true=true, stop_message="foo")
    m = SquareRooter(4)
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 1)
    @test !IC.done(c, state)
    @test v == [1, ]
    IC.train!(m, 1)
    state = IC.update!(c, m, 1, 2, state)
    @test IC.done(c, state)
    @test v == [1, 2]
    @test_logs(
        (:info, r"foo"),
        (@test IC.takedown(c, 1, state) == "foo"),
    )
end

@testset "Data" begin
    data = Float64[1.0, -0.9, 0]

    for option in [true, false]
        model = Particle(0.1)
        c = Data(data, stop_when_exhausted=option)

        state = IC.update!(c, model, 0, 1)
        IC.train!(model, 1)
        @test loss(model) ≈ 0.9
        @test IC.public_state(c, state) == (;)

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
            message = @test_logs (:info, IC.DATA_STOP) IC.takedown(c, 1, state)
            @test message == IC.DATA_STOP
        else
            state = @test_logs IC.update!(c, model, 0, 4, state)
            @test !IC.done(c, state)
            message = @test_logs IC.takedown(c, 1, state)
            @test message == ""
        end

    end
end

@testset "integration test" begin
    vector = repeat([-1, 1], outer=5);
    model = Particle(0.1)
    positions = Float64[]
    callback!(model) = (push!(positions, model.position); model.position)
    data, step, threshold, timelimit, info, callback = IC.train!(
        model,
        Data(vector),
        Step(5),
        Threshold(0.01),
        TimeLimit(0.0005),
        Info(IC.loss),
        Callback(callback!);
        verbosity=-1,
    )
    @test data.message == ""
    @test !data.triggered_stop
    @test step.new_steps isa Vector{Int}
    @test threshold.triggered_stop
    @test timelimit.table.time[end] < now()
    @test info.called_back isa Vector{Float64}
    @test callback.called_back isa Vector{Float64}
    @test loss(model) < 0.01

    model = Particle(0.1)
    positions = Float64[]
    # note: after data is exhausted, one gets one more control cycle.
    noise = fill((:info, ), length(vector) + 4)
    @test_logs(noise...,
               IC.train!(
                   model,
                   Data(vector, stop_when_exhausted=true),
                   Step(5),
                   WithNumberDo(),
                   WithLossDo(),
                   WithTrainingLossesDo(),
                   Threshold(0.01),
                   TimeLimit(0.0005),
                   Info(IC.loss),
                   Callback(callback!),
                   verbosity=1),
               )
    @test length(positions) == length(vector) + 1
    @test IC.loss(model) > 0.01
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
