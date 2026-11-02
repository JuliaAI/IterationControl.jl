using IterationControl
import IterationControl as IC

@testset "basic integration" begin
    m = IC.SquareRooter(4)
    controls = [Step(2), InvalidValue(), NumberLimit(3)]
    traces = []
    trace1, trace2, trace3 =
        IC.train!(m, controls...; verbosity=0, traces);

    @test traces == Any[trace1, trace2, trace3]
    @test trace1.table == (new_iterations = [2, 4, 6], new_steps = [1, 2, 3])
    @test !trace1.triggered_stop

    @test trace2.table == (; encountered = [nothing, nothing, nothing])
    @test !trace2.triggered_stop

    @test trace3.table == (; number = [1, 2, 3])
    @test trace3.triggered_stop

    m = IC.SquareRooter(4)
    @test_logs((:info, r"final loss"),
               (:info, r"final training loss"),
               (:info, r"Stop triggered by Num"),
               IC.train!(m, Step(2),  InvalidValue(), NumberLimit(3)));
    @test_logs((:info, r"Using these controls"),
               (:info, r"Stepping model for 2 more iterations"),
               (:info, r"Stepping model for 2 more iterations"),
               (:info, r"Stepping model for 2 more iterations"),
               (:info, r"final loss"),
               (:info, r"final training loss"),
               (:info, r"total control cycles \(steps\): 3"),
               (:info, r"Stop triggered by NumberLimit"),
               IC.train!(m, Step(2),
                         InvalidValue(),
                         NumberLimit(3);
                         verbosity=2),
               );

    # compact traces:
    m = IC.SquareRooter(4)
    controls = [Step(2), InvalidValue(), NumberLimit(3)]
    trace1, trace2, trace3 =
        IC.train!(m, controls...; verbosity=0, compact=true);
    @test trace1.new_iterations == [6,]
    @test trace2.encountered == [nothing,]
    @test trace3.number == [3,]

    controls[2] = Threshold()
    @test_throws IC.ERR_INCOMPATIBLE_TRACES IC.train!(m, controls; traces)
end

@testset "`traces` as kwarg" begin
    model = IterationControl.SquareRooter(4)
    controls = [Step(1), Threshold(0.01), Callback(_->sleep(0.1))]
    traces = []
    Threads.@spawn IterationControl.train!(model, controls...; verbosity=0, traces=traces)
    @test length(traces) == 0
    sleep(1)
    @test length(traces) == 3
    losses = deepcopy(traces[2].loss)
    @test length(losses) == 3
    @test losses[end] < 0.01 && losses[end] > 0.001

    # change a control and continue on with `traces=traces`:
    controls[2] = Threshold(1e-100)
    IterationControl.train!(model, controls...; verbosity=0, traces=traces)
    newlosses = traces[2].loss
    @test newlosses[1:3] == losses
    @test newlosses[end] < 1e-100
    @test traces[1].new_iterations == [1, 2, 3, 1, 2, 3]
end

struct WrappedSquareRooter
    s::IC.SquareRooter
end

IC.train!(w::WrappedSquareRooter, n::Int) = IC.train!(w.s, n)
IC.loss(w::WrappedSquareRooter) = IC.loss(w.s)
IC.expose(w::WrappedSquareRooter) = w.s

@testset "expose and Callback(..., raw=true)" begin

    wrapper = WrappedSquareRooter(IC.SquareRooter(4))

    roots = Any[]
    wrappers = Any[]

    function g(w)
        clone = deepcopy(w)
        push!(wrappers, clone)
        return clone
    end

    c1 = Callback(s->(push!(roots, s.root); s.root))
    c2 = Callback(g, raw=true)

    traces = IC.train!(wrapper, Step(1), c1, c2, NumberLimit(2), verbosity=0)

    @test map(w -> w.s.root, wrappers) ≈ roots
    @test traces[2].called_back == [2.5, 2.05]
end

@testset "skip integration" begin
    m = IC.SquareRooter(4)
    traces = IC.train!(
        m,
        Step(1),
        IterationControl.skip(WithNumberDo(), predicate=3),
        NumberLimit(10),
        verbosity=0,
    )
    @test traces[2].number == [3, 3, 3, 6, 6, 6, 9, 9]
end

@testset "integration test for skip" begin
    model = IterationControl.SquareRooter(4)
    @test_logs(
        (:info, r"final loss"),
        (:info, r"final training loss"),
        (:info, r"Stop triggered by"),
        IC.train!(
            model,
            Step(1),
            Threshold(2.1),
            WithNumberDo(),
            IterationControl.skip(WithLossDo(), predicate=3),
        ),
    )
end

@testset "integration test with PQ" begin
    # get complete training losses:
    model = IC.SquareRooter(1e40)
    IC.train!(model, 70)
    tlosses = IC.training_losses(model)

    # get complete losses:
    model = IC.SquareRooter(1e40)
    losses = map(1:70) do _
        IC.train!(model, 1)
        IC.loss(model)
    end

    # train with PQ criterion:
    K = 5
    model = IC.SquareRooter(1e40)
    pq = PQ(k=K)
    controls = [Step(1), NumberLimit(70), PQ(k=K)]
    _, _, trace = IC.train!(model, controls..., verbosity=0)

    # check internal state of PQ makes sense, as far as `loss` and
    # `training_losses` are concerned:
    loss0, _, _, training_loss, num_training_losses = trace.table
    @test loss0 == losses
    @test map(eachindex(loss0)) do j
        L = min(K, j) # what length of PQ training_losses should be
        num_training_losses[j] == L &&
            loss0[j] == losses[j] &&
            training_loss[j] == tlosses[j]
    end |> all
end

# https://github.com/JuliaAI/MLJIteration.jl/issues/36
@testset "integration test related to MLJIteration.jl #36" begin
    model = IC.SquareRooter(NaN)
    step_trace, invalid_trace, _ =
        IC.train!(model, Step(1), InvalidValue(), NumberLimit(3), verbosity=0)
    @test step_trace.new_iterations == [1,]
    @test occursin("Stopping", invalid_trace.message)
end
