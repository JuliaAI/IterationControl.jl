model = Particle()
withloss = WithLossDo(x-> nothing)
step = Step(1)

@test_throws IC.ERR_TRAIN IterationControl.train!(model)
@test_throws IC.err_train(model) IterationControl.train!(model, 1)

# lifting train!:
IC.train!(model::Particle, n) = train!(model, n)

@test_throws(IC.ERR_NEEDS_LOSS,
             IC.train!(model, step, withloss, NumberLimit(1)))

# lifting loss!:
IterationControl.loss(m::Particle) = loss(m)

@test_logs IC.train!(model, step, withloss, NumberLimit(1), verbosity=0)

@test_throws(IC.ERR_NEEDS_TRAINING_LOSSES,
             IC.train!(model, step, PQ(), NumberLimit(1)))

# lifting training_losses:
IterationControl.training_losses(m::Particle) = training_losses(m)

@test_logs IC.train!(model, Step(2), PQ(), NumberLimit(1), verbosity=0)

@test_throws(IC.err_ingest(model),
             IC.train!(model, Data(1:2), NumberLimit(1)))

#lifting ingest!:
IC.ingest!(model::Particle, datum) = ingest!(model, datum)

@test_logs IC.train!(model, Step(2), Data(1:1), NumberLimit(1), verbosity=0)
