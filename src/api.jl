# -------------------------------------------------------------------
# # MACHINE METHODS

# *models* are externally defined objects with certain functionality
# that is exposed by overloading these methods:

err_train(model) =
    ArgumentError("`IterationControl.train!(model, ::Int)` "*
                  "not overloaded "*
                  "for `typeof(model) = $(typeof(model))`. ")

err_ingest(model) =
    ArgumentError("Cannot use `Data` control here, as "*
                  "`IterationControl.ingest!(model)` "*
                  "not overloaded "*
                  "for `typeof(model) = $(typeof(model))`. ")

# ## COMPULSORY

# train `model` for another `Δn` iterations:
train!(model, Δn) = throw(err_train(model))


# ## REQUIRED FOR SOME CONTROL

# extract a single numerical estimate of `models`'s performance such
# as an out-of-sample loss; smaller is understood to be better:
loss(model) = nothing

# extract a vector of per-iteration "training" losses, accumulated in
# the last `train!(model, Δn)` call; this will have length `Δn`:
training_losses(model) = nothing

# ingest data:
ingest!(model, data) = throw(err_ingest(model))


# ## OPTIONAL

# so that user-specified functions (in eg `Info`) are applied to some
# internal part of the model, rather than `model` directly:
expose(model) = model

# switch for expose; not to be overloaded:
expose(model, raw) = expose(model, Val(raw))
expose(model, ::Val{true}) = model
expose(model, ::Val{false}) = expose(model)


# -------------------------------------------------------------------
# # CONTROL METHODS

# compulsory: `update!`
# optional: `done`, `takedown`

# called after first training event; returns initialized control
# "state":
update!(control, model, verbosity) = nothing

# called after all subsequent training events; returns new "state":
update!(control, model, verbosity, state) = state

# should we stop?
done(control, state) = false

# What to do after this control, or some other control, has triggered
# a stop. Returns user-inspectable outcomes associated with the
# control's applications (separate from logging). This should be a
# named tuple, except for composite controls which return a tuple of
# named tuples (see composite_controls.jl):
takedown(control, verbosity, state) = NamedTuple()

