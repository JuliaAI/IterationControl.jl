@testset "basic integration" begin
    m = SquareRooter(4)
    report = IC.train!(m, Step(2), InvalidValue(), NumberLimit(3); verbosity=0);
    @test report[1] == (Step(2), (new_iterations = 6,))
    @test report[2] == (InvalidValue(), (done=false, log=""))
    report[3] == (NumberLimit(3),
                  (done=true,
                   log="Stop triggered by NumberLimit(3) "*
                   "stopping criterion. "))

    m = SquareRooter(4)
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
               (:info, r"total control cycles\: 3"),
               (:info, r"A total of 6 iterations added"),
               (:info, r"Stop triggered by NumberLimit"),
               IC.train!(m, Step(2),
                         InvalidValue(),
                         NumberLimit(3);
                         verbosity=2));
end

struct WrappedSquareRooter
    s::SquareRooter
end

@testset "expose and Callback(..., raw=true)" begin
    IC.train!(w::WrappedSquareRooter, n) = IC.train!(w.s, n)
    IC.loss(w::WrappedSquareRooter) = IC.loss(w.s)
    IC.train!(w::WrappedSquareRooter, n) = IC.train!(w.s, n)
    IC.expose(w::WrappedSquareRooter) = w.s

    wrapper = WrappedSquareRooter(SquareRooter(4))

    roots = Any[]
    wrappers = Any[]

    g(w) = push!(wrappers, deepcopy(w))

    c1 = Callback(s->push!(roots, s.root))
    c2 = Callback(g, raw=true)

    IC.train!(wrapper, Step(1), c1, c2, NumberLimit(2), verbosity=0)

    @test map(w -> w.s.root, wrappers) ≈ roots
end

@testset "skip integration" begin
    m = SquareRooter(4)
    numbers = Int[]
    IC.train!(m,
              Step(1),
              IterationControl.skip(
                  WithNumberDo(x->push!(numbers, x)), predicate=3),
              NumberLimit(10), verbosity=0)
    @test numbers == [3, 6, 9]
end

@testset "integration test related to #38" begin
    model = IterationControl.SquareRooter(4)
    @test_logs((:info, r"number"),
               (:info, r"number"),
               (:info, r"final loss"),
               (:info, r"final training loss"),
               (:info, r"Stop triggered by"),
               IC.train!(model,
                         Step(1),
                         Threshold(2.1),
                         WithNumberDo(),
                         IterationControl.skip(WithLossDo(), predicate=3)))
end

@testset "integration test with PQ" begin
    # get complete training losses:
    model = SquareRooter(1e40)
    IC.train!(model, 70)
    tlosses = IC.training_losses(model)

    # get complete losses:
    model = SquareRooter(1e40)
    losses = map(1:70) do _
        IC.train!(model, 1)
        IC.loss(model)
    end

    # train with PQ criterion:
    K = 5
    model = SquareRooter(1e40)
    pq = PQ(k=K)
    states = Any[]
    f(s) = push!(states, s)
    controls = [Step(1), NumberLimit(70), IC.with_state_do(pq, f=f)]
    IC.train!(model, controls..., verbosity=0)

    # check internal state of PQ makes sense, as far as `loss` and
    # `training_losses` are concerned:
    @test map(s->s.loss, states) == losses
    @test map(enumerate(states)) do (j, s)
        L = min(K, j) # what length of PQ training_losses should be
        t= s.loss == losses[j] &&
            s.training_losses == reverse(tlosses[j-L+1:j])
    end |> all
end

# https://github.com/JuliaAI/MLJIteration.jl/issues/36
@testset "integration test related to MLJIteration.jl #36" begin
    model = SquareRooter(NaN)
    report =
        IC.train!(model, Step(1), InvalidValue(), NumberLimit(3), verbosity=0)
    @test report[1][2].new_iterations == 1
    @test occursin("Stopping", report[2][2].log)
end
