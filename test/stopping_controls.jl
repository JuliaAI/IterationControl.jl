@testset "loss getters" begin
    model=SquareRooter(4)
    @test IterationControl.get_loss(InvalidValue(), model) ==
        IterationControl.loss(model)
    @test_throws(IterationControl.err_getter(InvalidValue(), :loss, :junk),
                 IterationControl.get_loss(InvalidValue(), :junk))
    IterationControl.train!(model, 2)
    @test IterationControl.get_training_losses(PQ(), model) ==
        IterationControl.training_losses(model)
    @test_throws(
        IterationControl.err_getter(PQ(), :training_losses, :junk),
        IterationControl.get_training_losses(PQ(), :junk))
end

@testset "stopping criteria as controls" begin

    # A stopping criterion than ignores training losses:

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

    # I believe this will be fixed by
    # https://github.com/JuliaAI/EarlyStopping.jl/pull/29 
    @test_broken reverse(state.training_losses) == train_losses
    @test !IC.done(c, state)
    report = IC.takedown(c, 1, state)
    @test !report.done
    @test report.log == ""

end
