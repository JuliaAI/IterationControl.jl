@testset "debug" begin
    m = SquareRooter(4)
    test_controls = [Step(2), NotANumber(), GL(), Callback(println)]
    _info = fill((:info, r""), 2*length(test_controls))

    @test_logs(_info...,
               for c in [Step(2), NotANumber(), GL(), Callback()]
               d = IC.debug(c)
               state = IC.update!(c, m, 0)
               @test state == IC.update!(d, m, 0)
               state_c = IC.update!(c, m, 0, state)
               state_d =  IC.update!(d, m, 0, state)
               @test state_c == state_c
               @test IC.done(c, state_c) == IC.done(d, state_d)
               end)
end

@testset "skip" begin
    m = SquareRooter(4)
    test_controls = [Step(2), NotANumber(), GL(), Callback(println)]
    _info = fill((:info, r""), 2)

    @test_logs(_info...,
               for c in [Step(2), NotANumber(), GL(), Callback()]
               s = IC.skip(c, predicate=2)
               @test !s.predicate(0)
               @test s.predicate(1)
               @test !s.predicate(2)
               atomic_state = IC.update!(c, m, 0)
               state = IC.update!(s, m, 0)
               @test state == (n = 1, )
               state =  IC.update!(s, m, 0, state)
               @test state == (atomic_state = atomic_state, n = 2)
               state =  IC.update!(s, m, 0, state)
               @test state == (atomic_state = atomic_state, n = 3)
               atomic_state = IC.update!(c, m, 0, atomic_state)
               state =  IC.update!(s, m, 0, state)
               @test state == (atomic_state = atomic_state, n = 4)
               @test IC.done(c, atomic_state) == IC.done(s, state)
               @test IC.takedown(c, 0, atomic_state) == IC.takedown(s, 0, state)
               end)
end
