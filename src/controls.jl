# # STOPPING CRIITERIA, AS CONTROLS

# `StoppingCriterion`objects are defined in EarlyStopping.jl

_err_loss() = error("`IterationControl.loss` not suitably overloaded "*
              "for $(typeof(model)). ")

# initialization:
function update!(c::StoppingCriterion,
                model,
                verbosity)
    _loss = loss(model)
    _loss == nothing && _err_loss()
    _training_losses = training_losses(model)
    if _training_losses === nothing || isempty(_training_losses)
        state = EarlyStopping.update(c, _loss)
    else # first consume all training losses, then update! loss:
        state = EarlyStopping.update_training(c, first(_training_losses))
        for tloss in _training_losses[2:end]
            state = EarlyStopping.update_training(c, tloss, state)
        end
        state = EarlyStopping.update(c, _loss, state)
    end
    return state
end

# regular update!:
function update!(c::StoppingCriterion,
                model,
                verbosity,
                state)
    _loss = loss(model)
    _loss == nothing && _err_loss()
    _training_losses = training_losses(model)
    if _training_losses === nothing || isempty(_training_losses)
        state = EarlyStopping.update(c, _loss, state)
    else # first consume all training losses, then update! loss:
        for tloss in _training_losses
            state = EarlyStopping.update_training(c, tloss, state)
        end
        state = EarlyStopping.update(c, _loss, state)
    end
    return state
end

function takedown(c::StoppingCriterion, verbosity, state)
    if done(c, state)
        message =  EarlyStopping.message(c, state)
        verbosity > 0 && @info message
        return (done = true, log = message)
    else
        return (done = false, log = "")
    end
end


# # TRAIN

struct Train
    n::Int
end

# constructor:
Train(; n=5) = Train(n)

@create_docs(Train,
             header="Train(; n=1)",
             example="Train(2)",
             body="Train the model for `n` iterations. "*
             "Will never trigger a stop. ")

function update!(c::Train, model, verbosity, args...)
    if verbosity > 0
        @info "Training model for $(c.n) iterations. "
    else
        nothing
    end
    train!(model, c.n)
end

# # Info

struct Info{F<:Function}
    f::F
end

# constructor:
Info(; f::Function=identity) = Info(f)

@create_docs(Info,
             header="Info(f=identity)",
             example="Info()",
             body="Log at the `Info` level the value of `f(model)`, "*
             "where `model` "*
             "is the object being iterated. If "*
             "`IterativeControl.expose(model)` has been overloaded, then "*
             "log `f(expose(model))` instead.\n\n"*
             "Can be suppressed by setting the global verbosity level "*
             "sufficiently low. \n\n"*
             "See also [`Warn`](@ref), [`Error`](@ref). ")

function update!(c::Info, model, verbosity, args...)
    verbosity < 0 || @info _log_eval(c.f, model)
    return nothing
end


# # WARN

struct Warn{P<:Function,F<:Union{Function,String}}
    predicate::P
    f::F
end

# constructor:
Warn(predicate; f="") = Warn(predicate, f)

@create_docs(Warn,
             header="Warn(predicate; f=(_ -> \"\"))",
             example="Warn(m->length(m.cache) > 100, f=\"Memory low\")",
             body="Log at the `Warn` level the value of `f(model)` "*
             "(or just `f` if `f` is a string) "*
             "whenever `predicate(model)` is `true`. Here `model` "*
             "is the object being iterated. `If "*
             "`IterativeControl.expose(model)` has been overloaded, then "*
             "log `f(expose(model))` instead.\n\n"*
             "Can be suppressed by setting the global verbosity level "*
             "sufficiently low.\n\n"*
             "See also [`Info`](@ref), [`Error`](@ref). ")

function update!(c::Warn, model, verbosity, args...)
    verbosity > 0 && c.predicate(model) &&
        @warn _log_eval(c.f, model)
    return nothing
end

function update!(c::Warn, model, verbosity, warnings=())
    if c.predicate(model)
        warning = _log_eval(c.f, model)
        verbosity < 0 || @warn warning
        state = tuple(warnings..., warning)
    else
        state = warnings
    end
    return state
end

takedown(c::Warn, verbosity, state) = (warnings = state,)


# # ERROR

struct Error{P<:Function,F<:Union{Function,String}}
    predicate::P
    f::F
    exception::Union{Exception, Nothing}
end

# constructor:
Error(predicate; f="", exception=nothing) = Error(predicate, f, exception)

@create_docs(Error,
             header="Error(predicate; f=\"\"))",
             example="Error(m->length(m.cache) > 100, f=\"Memory low\")",
             body="Log at the `Error` level the value of `f(model)` "*
             "(or just `f` if `f` is a string) "*
             "whenever `predicate(model)` is `true`, in which case "*
             "stop iteration early. Here `model` "*
             "is the object being iterated. `If "*
             "`IterativeControl.expose(model)` has been overloaded, then "*
             "log `f(expose(model))` instead.\n\n"*
             "Specify `exception=...` to throw an execption.\n\n"*
             "See also [`Info`](@ref), [`Warn`](@ref). ")

function update!(c::Error,
                 model,
                 verbosity,
                 state=(done = false, error=()))
    if c.predicate(model)
        error = _log_eval(c.f, model)
        @error error
        c.exception isa Nothing || throw(c.exception)
        state = (done = true, error=error)
    end
    return state
end

done(c::Error, state) = state.done

takedown(c::Error, verbosity, state) = state


# # Callback

struct Callback{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
Callback(f::Function;
         stop_if_true=false,
         stop_message=nothing) = Callback(f, stop_if_true, stop_message)
Callback(; f=identity, kwargs...) = Callback(f, kwargs...)

@create_docs(Callback,
             header="Callback(f= _->nothing, stop_if_true=false, "*
             "stop_message=nothing)",
             example="Callback(m->put!(v, loss(m))",
             body="Call `f(model)`, or `f(expose(model))` if "*
             "`IterativeControl.expose(model)` has been overloaded. "*
             "If `stop_if_true` is `true`, then trigger an early stop "*
             "if the value returned by `f` is `true`, logging the "*
             "`stop_message` if specified. ")

function update!(c::Callback, model, verbosity, state=(done=false, ))
    r = c.f(expose(model))
    done = (c.stop_if_true && r isa Bool && r) ? true : false
    return (done=done,)
end

done(c::Callback, state) = state.done

function takedown(c::Callback, verbosity, state)
    if state.done
        message = c.stop_message isa Nothing ?
            "Stopping early stop triggered by a `Callback` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return (done = true, log = message)
    else
        return (done = false, log = "")
    end
end