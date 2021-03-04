model = WhatsNext()

@test_throws IC.ERR_TRAIN IterationControl.train!(model)
@test_throws IC.err_train(model)  IterationControl.train!(model, 1)

# lifting train!:
IC.train!(model::WhatsNext, n) = train!(model, n)

@test_throws(IC.err_loss(NotANumber(), model),
             IC.train!(model, NotANumber(), NumberLimit(1)))

# lifting loss!:
IterationControl.loss(m::WhatsNext) = abs(m.position)

@test_throws(IC.err_training_losses(PQ(), model),
             IC.train!(model, PQ(), NumberLimit(1)))

# lifting training_losses:
IterationControl.training_losses(m::WhatsNext) = m.training_losses

##### now do the data stuff
