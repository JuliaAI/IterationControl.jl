model = Particle()
invalid = InvalidValue()

@test_throws IC.ERR_TRAIN IterationControl.train!(model)
@test_throws IC.err_train(model)  IterationControl.train!(model, 1)

# lifting train!:
IC.train!(model::Particle, n) = train!(model, n)

@test_throws(IC.err_getter(invalid, :loss, model),
             IC.train!(model, invalid, NumberLimit(1)))

# lifting loss!:
IterationControl.loss(m::Particle) = loss(m)

IC.train!(model, invalid, NumberLimit(1), verbosity=0)

@test_throws(IC.err_getter(PQ(), :training_losses, model),
             IC.train!(model, PQ(), NumberLimit(1)))

# lifting training_losses:
IterationControl.training_losses(m::Particle) = training_losses(m)

IC.train!(model, Step(2), PQ(), NumberLimit(1), verbosity=0)

@test_throws(IC.err_ingest(model),
             IC.train!(model, Data(1:2), NumberLimit(1)))

#lifting ingest!:
IC.ingest!(model::Particle, datum) = ingest!(model, datum)

IC.train!(model, Data(1:1), NumberLimit(1), verbosity=0);
