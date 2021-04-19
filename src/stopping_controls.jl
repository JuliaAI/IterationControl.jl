# # STOPPING CRIITERIA, AS CONTROLS

# `StoppingCriterion`objects are defined in EarlyStopping.jl


# ## LOSS GETTERS

# `get_loss(control, model)` throws an error if control needs
# `IC.loss` overloaded for `type(model)` and it has not been so
# overloaded. If `control` does not need `IC.loss`, then `nothing` is
# returned. In the other cases, the sought after loss is
# returned. `get_training_losses` is similarly defined.

err_getter(c, f, model) =
    ArgumentError("Use of `$c` control here requires that "*
                  "`IterationControl.$f(model)` be "*
                  "overloaded for `typeof(model)=$(typeof(model))`. ")

for f in [:loss, :training_losses]
    g = Symbol(string(:get_, f))
    t = Symbol(string(:needs_, f))
    fstr = string(f)
    eval(quote
         $g(c, model) = $g(c, model, Val(ES.$t(c)))
         $g(c, model, ::Val{false}) = nothing
         @inline function $g(c, model, ::Val{true})
             it = $f(model)
             it isa Nothing && throw(err_getter(c, $fstr, model))
             return it
         end
         end)
end


# ## API IMPLEMENTATION

function update!(c::StoppingCriterion,
                 model,
                 verbosity,
                 n)
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
                 n,
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


