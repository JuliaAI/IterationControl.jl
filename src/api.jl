# -------------------------------------------------------------------
# # MODEL METHODS

err_train(model) =
    ArgumentError("`IterationControl.train!(model, ::Int)` "*
                  "not overloaded "*
                  "for `typeof(model) = $(typeof(model))`. ")

err_ingest(model) =
    ArgumentError("Cannot use `Data` control here, as "*
                  "`IterationControl.ingest!(model, datum)` "*
                  "not overloaded "*
                  "for `typeof(model) = $(typeof(model))`. ")



# ## COMPULSORY METHOD TO LIFT MODEL TRAINING TO IterationControl

# train `model` for another `Δn` iterations:

train!(model, Δn::Int) = throw(err_train(model))


# ## OTHER METHODS THAT LIFT TO IterationControl A MODEL METHOD

# extract a single numerical estimate of `models`'s performance such
# as an out-of-sample loss; smaller is understood to be better:
"""
    IterationControl.loss(model)

Return the current loss associated with `model`, which will be some measure of the quality
of `model`, such as an out-of-sample statistical loss. The smaller the loss, the better
the inferred quality.

This method must be overloaded for any `model` type wanting to support an iteration
`control` with `IterationControl.needs_loss(control) == true`.

"""
loss(model) = nothing

# extract a vector of per-iteration "training" losses, accumulated in
# the last `train!(model, Δn)` call; this will have length `Δn`:

"""
    IterationControl.training_losses(model)

Return a vector of all, or recent, per-iteration "training" losses, since training of
`model` began. Typically, a training loss is an in-sample statistical measure of the
quality of `model`, as opposed to the (out-of-sample) loss returned by
`IterationControl.loss(model)`.  The smaller the loss, the better the inferred quality.

Training losses are returned in chronological order. 

This method must be overloaded for any `model` type wanting to support an iteration
`control` with `IterationControl.needs_training_losses(control) == true`.

"""
training_losses(model) = nothing

"""
    IterationControl.ingest!(model, datum)

Have `model` ingest new training `datum`, without actually training `model`.

This method must be overloaded for any `model` type wanting to support an iteration
`control` with `IterationControl.needs_ingest(control) == true`.

"""
ingest!(model, data) = throw(err_ingest(model))


"""
    IterationControl.expose(model)

Unwrap `model` to expose that part on which user-specified functions in iteration controls
should act. For example, in [`IterationControl.train!`](@ref)`(model, Step(),
Info(model->print(model))`, it is `IterationControl.expose(model)` that gets logged
to `Info`, not `model`.

"""
expose(model) = model

# switch for expose; not to be overloaded:
expose(model, use_wrapped) = expose(model, Val(use_wrapped))
expose(model, ::Val{true}) = model
expose(model, ::Val{false}) = expose(model)


# # INTERFACE IMPLEMENTED BY ITERATION CONTROLS

# compulsory: `update!`
# optional: `done`, `takedown`, `needs_training_losses`

const DOC_UPDATE = """
    The returned state, `state`, is not part of the public API. To inspect a user-facing form
    of `state`, as recorded in control traces, call [`IterationControl.public_state(control,
    state)`](@ref).

    # New control implementations
    """

"""
    IterationControl.update!(control, model, verbosity, n)

Return intialized internal state for `control`, when controlling `model`. Here `n` is the
current number of training iterations that have have been applied to `model` so far.  (This
needn't be one, as a previously applied `Step(k)` control may have `k > 1`).

$DOC_UPDATE Compulsory for all new `control` types.

"""
update!(control, model, verbosity, n) = nothing

"""
    IterationControl.update!(control, model, verbosity, n, old_state)

Return updated internal state for `control`, when controlling `model`. Here `n` is the
current number of training iterations that have have been applied to `model` so far.

$DOC_UPDATE May mutate `state`, but typically does not.
A fallback returns `old_state`.
"""
update!(control, model, verbosity, n, state) = state


"""
    IterationControl.done(control, state)

Returns `true` if and only if the `control` wants to trigger a stop. Method belongs to
`EarlyStopping.jl`, a dependency of `IterationControl`.

# New control implementations

A fallback always returns `false`.
"""
EarlyStopping.done

"""
    takedown(control, verbosity, state) -> message

With the understanding that all `control` updates are complete, log to `Info` or `Warn` a
relevant message, and return the message as a string.

# New control implementations

Typically the message is only logged to `Info` if non-empty and `verbosity > 0` or, if it
is a warning, logged to `Warn` if `verbosity > -1`. Irrespective of the `verbosity` level,
the message should be returned, but can be `""`, the fallback value.

Ideally, the message should be brief. (The user can get detailed information about the
final state of the control by inspecting the corresponding [`ControlTrace`](@ref) object.)

"""
takedown(control, verbosity, state) = ""

"""
    IterationControl.needs_training_losses(control)

Returns `true` if `control` is only supported for a model for which
[`IterationControl.training_losses(model)`](@ref) is implemented.

"""
needs_training_losses(control) = false

"""
    IterationControl.needs_loss(control)

Returns `true` if `control` is only supported for a model for which
[`IterationControl.loss(model)`](@ref) is implemented.

"""
needs_loss(control) = false

"""
    IterationControl.needs_ingest(control)

Returns `true` if `control` is only supported for a model for which
[`IterationControl.ingest!(model, datum)`](@ref) is implemented.

"""
needs_ingest(control) = false
