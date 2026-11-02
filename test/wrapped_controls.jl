using IterationControl
import IterationControl as IC
import IterationControl.SquareRooter

@testset "louder" begin
    m = SquareRooter(4)
    control = Warn(_->true, f="42")

    c = IC.louder(control, by=0)
    IC.train!(m, 1)
    state = @test_logs IC.update!(c, m, -1, 1)
    IC.train!(m, 2)
    state = @test_logs (:warn, r"42") IC.update!(c, m, 0, 2, state)
    @test_logs(
        (:warn, r"A "),
        (@test IC.takedown(c, 1, state) ==
            "A `Warn` control issued 2 warnings."),
    )
    @test_logs(
        (:warn, r"A "),
        (@test IC.takedown(c, 0, state) ==
            "A `Warn` control issued 2 warnings."),
    )
    @test_logs(
        (@test IC.takedown(c, -1, state) ==
            "A `Warn` control issued 2 warnings."),
    )

    c = IC.louder(control, by=1)
    IC.train!(m, 1)
    state = @test_logs IC.update!(c, m, -2, 1)
    IC.train!(m, 2)
    state = @test_logs (:warn, r"42") IC.update!(c, m, -1, 2, state)
    @test_logs(
        (:warn, r"A "),
        (@test IC.takedown(c, -1, state) ==
            "A `Warn` control issued 2 warnings."),
    )
    @test_logs IC.takedown(c, -2, state)
end

@testset "with_state_do" begin
    m = SquareRooter(4)
    test_controls = [Step(2), InvalidValue(), Callback()]
    _info = fill((:info, r""), 2*length(test_controls))

    @test_logs(_info...,
               for c in test_controls
               d = IC.with_state_do(c)
               state = IC.update!(c, m, 1, 1)
               @test state == IC.update!(d, m, 1, 1)
               state_c = IC.update!(c, m, 0, 2, state)
               state_d =  IC.update!(d, m, 0, 2, state)
               @test state_c == state_d
               @test IC.done(c, state_c) == IC.done(d, state_d)
               end)

    # integration test:
    m = SquareRooter(4)
    v = Any[]
    f(public_state) = push!(v, public_state)
    step, numberlimit =
        IC.train!(m, IC.with_state_do(Step(2), f=f), NumberLimit(10), verbosity=0)
    @test v[end] == (new_iterations=20, new_steps=10)
    @test length(v) == 10
end

@testset "skip" begin

    # constructor and display:
    s = IC.skip(Threshold(0.1), predicate=2)
    @test s.predicate(2)
    @test !s.predicate(3)
    @test sprint(show, s) == "skip(Threshold(0.1), predicate=2)"
    s = IC.skip(Threshold(0.1), predicate=IC.accept(2))
    @test s.predicate(2)
    @test !s.predicate(3)
    @test sprint(show, s) == "skip(Threshold(0.1), …)"
    @test sprint(show, MIME("text/plain"), s) == "skip(Threshold(0.1), …)"

    m = SquareRooter(4)
    test_controls = [Step(2), InvalidValue(), Callback(), Threshold(0.1)]

    for c in test_controls
        s = IC.skip(c, predicate=2)
        @test !s.predicate(1)
        @test s.predicate(2)
        @test !s.predicate(3)
        atomic_state = IC.update!(c, m, 0, 1)
        state = IC.update!(s, m, 0, 1)
        @test state == (n = 1, )
        @test isnothing(IC.public_state(s, state))
        state =  IC.update!(s, m, 0, 2, state)
        @test state == (atomic_state = atomic_state, n = 2)
        state =  IC.update!(s, m, 0, 3, state)
        @test state == (atomic_state = atomic_state, n = 3)
        atomic_state = IC.update!(c, m, 0, 2, atomic_state)
        state =  IC.update!(s, m, 0, 4, state)
        @test state == (atomic_state = atomic_state, n = 4)
        @test IC.done(c, atomic_state) == IC.done(s, state)
        @test IC.takedown(c, 0, atomic_state) == IC.takedown(s, 0, state)
    end
end
