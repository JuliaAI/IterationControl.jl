model = IterationControl.SquareRooter(4.0)
IterationControl.train!(model, 1)
@test model.root ≈ 2.5
@test IterationControl.loss(model) ≈ 25/4 - 4
IterationControl.train!(model, 100)
@test IterationControl.loss(model) ≈ 0
@test IterationControl.training_losses(model)[1:2] ≈
    abs.([41/20 - 5/2, 3281/1640 - 41/20])

true
