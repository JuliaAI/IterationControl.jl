@testset "stopping criteria as controls" begin

    # A stopping criterion that ignores training losses:

    m = SquareRooter(4)
    c = NumberLimit(2)

    IC.train!(m, 3)
    state = IC.update!(c, m, 0, 1)
    @test state == 1
    @test !IC.done(c, state)
    IC.train!(m, 3)
    state = IC.update!(c, m, 0, 2, state)
    @test state == 2
    @test IC.done(c, state)
    report = @test_logs (:info, r"NumberLimit\(2\)") IC.takedown(c, 1, state)
    @test report.done
    @test report.log ==
        "Stop triggered by NumberLimit(2) stopping criterion. "

    # A stopping criterion than uses training losses:

    m = SquareRooter(4)
    c = PQ(k=5)

    # Note that `SquareRooter` does not cache training losses. It only
    # makes the most recent losses available, which is the worst-case
    # scenario.

    IC.train!(m, 3)
    state = IC.update!(c, m, 0, 1)
    train_losses = m.training_losses
    @test reverse(state.training_losses) == train_losses # length 3
    @test !IC.done(c, state)

    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 2, state)
    train_losses = vcat(train_losses, m.training_losses) # length 2+3 = 5

    @test reverse(state.training_losses) == train_losses
    @test !IC.done(c, state)
    report = IC.takedown(c, 1, state)
    @test !report.done
    @test report.log == ""
end
