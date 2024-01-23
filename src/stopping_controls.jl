# # STOPPING CRIITERIA, AS CONTROLS

# `StoppingCriterion`objects are defined in EarlyStopping.jl

# non-wrapping stopping criteria that are nonsensical to apply if
# `IterationControl.training_losses(model)` is not overloaded:
const ATOMIC_CRIITERIA_NEEDING_TRAINING_LOSSES = [:PQ, ]
const ATOMIC_CRIITERIA_NEEDING_LOSS = [:Threshold, :GL, :PQ, :Patience]

# stopping criterion that wrap a single stopping criterion (must have
# `:criterion` as a field):
const EARLY_STOPPING_WRAPPERS = [:Warmup, ]


for ex in ATOMIC_CRIITERIA_NEEDING_LOSS
    quote
        needs_loss(::$ex) = true
    end |> eval
end

for ex in ATOMIC_CRIITERIA_NEEDING_TRAINING_LOSSES
    quote
        needs_training_losses(::$ex) = true
    end |> eval
end

for ex in EARLY_STOPPING_WRAPPERS
    quote
        needs_loss(wrapper::$ex) =
            needs_loss(wrapper.criterion)
        needs_training_losses(wrapper::$ex) =
            needs_training_losses(wrapper.criterion)
    end |> eval
end

function update!(c::StoppingCriterion,
                 model,
                 verbosity,
                 n, state=nothing)
    _loss = loss(model)
    _training_losses = training_losses(model)
    if _training_losses !== nothing && !isempty(_training_losses)
        for tloss in _training_losses
            state = ES.update_training(c, tloss, state)
        end
    end
    state = ES.update(c, _loss, state)
    return state
end

function takedown(c::StoppingCriterion, verbosity, state)
    if done(c, state)
        message =  ES.message(c, state)
        verbosity > 0 && @info message
        return (done = true, log = message)
    else
        return (done = false, log = "")
    end
end
