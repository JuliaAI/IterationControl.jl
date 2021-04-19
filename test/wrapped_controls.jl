@testset "louder" begin
    m = SquareRooter(4)
    control = Warn(_->true, f="42")

    c = IC.louder(control, by=0)
    IC.train!(m, 1)
    state = @test_logs IC.update!(c, m, -1, 1)
    IC.train!(m, 2)
    state = @test_logs (:warn, r"42") IC.update!(c, m, 0, 2, state)
    @test_logs IC.takedown(c, 1, state)
    @test_logs (:warn, r"A ") IC.takedown(c, 2, state)

    c = IC.louder(control, by=1)
    IC.train!(m, 1)
    state = @test_logs IC.update!(c, m, -2, 1)
    IC.train!(m, 2)
    state = @test_logs (:warn, r"42") IC.update!(c, m, -1, 2, state)
    @test_logs IC.takedown(c, 0, state)
    @test_logs (:warn, r"A ") IC.takedown(c, 1, state)
end

@testset "debug" begin
    m = SquareRooter(4)
    test_controls = [Step(2), InvalidValue(), GL(), Callback(println)]
    _info = fill((:info, r""), 2*length(test_controls))

    @test_logs(_info...,
               for c in [Step(2), InvalidValue(), GL(), Callback()]
               d = IC.debug(c)
               state = IC.update!(c, m, 1, 1)
               @test state == IC.update!(d, m, 1, 1)
               state_c = IC.update!(c, m, 0, 2, state)
               state_d =  IC.update!(d, m, 0, 2, state)
               @test state_c == state_c
               @test IC.done(c, state_c) == IC.done(d, state_d)
               end)
end

@testset "skip" begin
    m = SquareRooter(4)
    test_controls = [Step(2), InvalidValue(), GL(), Callback(println)]
    _info = fill((:info, r""), 2)

    @test_logs(_info...,
               for c in [Step(2), InvalidValue(), GL(), Callback()]
               s = IC.skip(c, predicate=2)
               @test !s.predicate(1)
               @test s.predicate(2)
               @test !s.predicate(3)
               atomic_state = IC.update!(c, m, 1, 1)
               state = IC.update!(s, m, 1, 1)
               @test state == (n = 1, )
               state =  IC.update!(s, m, 1, 2, state)
               @test state == (atomic_state = atomic_state, n = 2)
               state =  IC.update!(s, m, 1, 3, state)
               @test state == (atomic_state = atomic_state, n = 3)
               atomic_state = IC.update!(c, m, 1, 2, atomic_state)
               state =  IC.update!(s, m, 1, 4, state)
               @test state == (atomic_state = atomic_state, n = 4)
               @test IC.done(c, atomic_state) == IC.done(s, state)
               @test IC.takedown(c, 0, atomic_state) == IC.takedown(s, 0, state)
               end)
end
