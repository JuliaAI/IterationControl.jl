# -------------------------------------------------------------------
# # MODEL METHODS

# *models* are externally defined objects with certain functionality
# that is exposed by overloading these methods:

err_train(model) =
    ArgumentError("`IterationControl.train!(model, ::Int)` "*
                  "not overloaded "*
                  "for `typeof(model) = $(typeof(model))`. ")

err_ingest(model) =
    ArgumentError("Cannot use `Data` control here, as "*
                  "`IterationControl.ingest!(model, datum)` "*
                  "not overloaded "*
                  "for `typeof(model) = $(typeof(model))`. ")

# ## COMPULSORY

# train `model` for another `Δn` iterations:
train!(model, Δn) = throw(err_train(model))


# ## REQUIRED FOR SOME CONTROLS

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
# optional: `done`, `takedown`, `needs_training_losses`

# called after first training event; returns initialized control
# "state":
update!(control, model, verbosity, n) = nothing

# called after all subsequent training events; returns new "state":
update!(control, model, verbosity, n, state) = state

# should we stop? This method is imported from EarlyStopping with this default:
# done(control, state) = false

# What to do after `control`, or some other control, has triggered a stop. Returns a short
# (possible empty) message string, which it should log to `:info` if it's non-empty and
# `verbosity > 0`. Ideally the message should fit on one line. (Detailed information can
# be accessed with `EarlyStopping.public_state(control, final_state)`.)
takedown(control, verbosity, state) = ""

# If it is nonsensical to apply `control` to any model for which `training_losses(model)`
# has not been overloaded and we want an error thrown when this is attempted, then
# overload this trait to take the value `true`. Otherwise `control` is applied anyway, and
# `training_losses`, if called, returns `nothing`.
needs_training_losses(control) = false

# If it is nonsensical to apply `control` to any model for which `loss(model)` has not
# been overloaded and we want an error thrown when this is attempted, then overload this
# trait to take the value `true`. Otherwise `control` is applied anyway, and `loss`, if
# called, returns `nothing`.
needs_loss(control) = false
