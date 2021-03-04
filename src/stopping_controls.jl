# # STOPPING CRIITERIA, AS CONTROLS

# `StoppingCriterion`objects are defined in EarlyStopping.jl


# ## LOSS GETTERS

for f in [:loss, :training_losses]
    g = Symbol(string(:get_, f))
    e = Symbol(string(:err_, f))
    t = Symbol(string(:needs_, f))
    eval(quote
         $e(c, model) =
             ArgumentError("Use of `$c` control here requires that "*
                           "`IterationControl.loss(model)` be "*
                           "overloaded for `typeof(model)=$(typeof(model))`. ")

         $g(c, model) = $g(c, model, Val(ES.$t(c)))
         $g(c, model, ::Val{false}) = nothing
         @inline function $g(c, model, ::Val{true})
             it = $f(model)
             it isa Nothing && throw($e(c, model))
             return it
         end
         end)
end


# ## API IMPLEMENTATION

function update!(c::StoppingCriterion,
                model,
                verbosity)
    _loss = get_loss(c, model)
    _training_losses = get_training_losses(c, model)
    if _training_losses === nothing || isempty(_training_losses)
        state = ES.update(c, _loss)
    else # first consume all training losses, then update! loss:
        state = ES.update_training(c, first(_training_losses))
        for tloss in _training_losses[2:end]
            state = ES.update_training(c, tloss, state)
        end
        state = ES.update(c, _loss, state)
    end
    return state
end

# regular update!:
function update!(c::StoppingCriterion,
                model,
                verbosity,
                state)
    _loss = get_loss(c, model)
    _training_losses = get_training_losses(c, model)
    if _training_losses === nothing || isempty(_training_losses)
        state = ES.update(c, _loss, state)
    else # first consume all training losses, then update! loss:
        for tloss in _training_losses
            state = ES.update_training(c, tloss, state)
        end
        state = ES.update(c, _loss, state)
    end
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


