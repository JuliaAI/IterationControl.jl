@testset "basic integration" begin
    m = SquareRooter(4)
    report = IC.train!(m, Step(2),  NotANumber(), NumberLimit(3); verbosity=0);
    @test report[1] == (Step(2), NamedTuple())
    @test report[2] == (NotANumber(), (done=false, log=""))
    report[3] == (NumberLimit(3),
                  (done=true,
                   log="Stop triggered by NumberLimit(3) "*
                   "stopping criterion. "))

    m = SquareRooter(4)
    @test_logs((:info, r"Stop triggered by Num"),
               IC.train!(m, Step(2),  NotANumber(), NumberLimit(3)));
    @test_logs((:info, r"Using these controls"),
               (:info, r"Stepping model for 2 iterations"),
               (:info, r"Stepping model for 2 iterations"),
               (:info, r"Stepping model for 2 iterations"),
               (:info, r"Stop triggered by NumberLimit"),
               IC.train!(m, Step(2),
                         NotANumber(),
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
              NumberLimit(10))
    @test numbers == [1, 2, 3]
end
