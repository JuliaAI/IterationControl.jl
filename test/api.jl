model = Particle()

@test_throws IC.ERR_TRAIN IterationControl.train!(model)
@test_throws IC.err_train(model)  IterationControl.train!(model, 1)

# lifting train!:
IC.train!(model::Particle, n) = train!(model, n)

@test_throws(IC.err_loss(NotANumber(), model),
             IC.train!(model, NotANumber(), NumberLimit(1)))

# lifting loss!:
IterationControl.loss(m::Particle) = abs(m.position)

IC.train!(model, NotANumber(), NumberLimit(1))

@test_throws(IC.err_training_losses(PQ(), model),
             IC.train!(model, PQ(), NumberLimit(1)))

# lifting training_losses:
IterationControl.training_losses(m::Particle) = training_losses(m)

IC.train!(model, Train(2), PQ(), NumberLimit(1))

@test_throws(IC.err_ingest(model),
             IC.train!(model, Data(1:2), NumberLimit(1)))

#lifting ingest!:
IC.ingest!(model::Particle, datum) = ingest!(model, datum)

IC.train!(model, Data(1:1), NumberLimit(1));
