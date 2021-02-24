@testset "constructors, IC.flat" begin
    @test IC.composite() == Never()
    c1 = Patience(1)
    c2 = NotANumber()
    c3 = Train(1)

    @test IC.composite(c1) == c1

    d = IC.composite(c1, c2, Never(), c3, c1)
    show(d)
    @test IC._flat(d) == (c1, c2, c3)

    # codecov:
    @test IC._in(Never(), c1)
    @test IC._in(Never(), d)
end

@testset "behaviour" begin
    a = Train(4)
    b = NumberLimit(2)
    c = NotANumber()
    d = IC.composite(a, b, c)

    # separated:
    m = SquareRooter(4)
    state_a1 = IC.update!(a, m, 0)
    state_b1 = IC.update!(b, m, 0)
    state_c1 = IC.update!(c, m, 0)
    state_a2 = IC.update!(a, m, 0, state_a1)
    state_b2 = IC.update!(b, m, 0, state_b1)
    state_c2 = IC.update!(c, m, 0, state_c1)
    done_a = IC.done(a, state_a2)
    done_b = IC.done(b, state_b2)
    done_c = IC.done(c, state_c2)
    report_a = IC.takedown(a, 0, state_a2)
    report_b = IC.takedown(b, 0, state_b2)
    report_c = IC.takedown(c, 0, state_c2)

    # composed:
    m = SquareRooter(4)
    state_d1 = IC.update!(d, m, 0)
    @test IC.flat(state_d1) == (state_a1, state_b1, state_c1)
    state_d2 = IC.update!(d, m, 0, state_d1)
    @test IC.flat(state_d2) == (state_a2, state_b2, state_c2)
    done_d = IC.done(d, state_d2)
    @test done_d == done_a || done_b || done_c
    report_d = IC.takedown(d, 0, state_d2)
    @test report_d == ((a, report_a), (b, report_b), (c, report_c))

end

true
