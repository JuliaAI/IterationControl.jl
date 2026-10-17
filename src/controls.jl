# # Step

struct Step
    n::Int
end

# constructor:
Step(; n=1) = Step(n)

@create_docs(Step,
             header="Step(; n=1)",
             example="Step(2)",
             body="Train for `n` more iterations. "*
             "Will never trigger a stop. ")

function update!(c::Step, model, verbosity, n, state=(new_iterations = 0,))
    new_iterations = state.new_iterations
    verbosity > 1 &&
        @info "Stepping model for $(c.n) more iterations. "
    train!(model, c.n)
    state  = (new_iterations = new_iterations + c.n,)
    return state
end
public_state(c::Step, state) = (; state..., new_steps=div(state.new_iterations, c.n))


# # Info

struct Info{F<:Function}
    f::F
end

# constructor:
Info(; f::Function=identity) = Info(f)

@create_docs(Info,
             header="Info(f=identity)",
             example="Info(my_loss_function)",
             body="Log to `Info` the value of `f(m)`, "*
             "where `m` "*
             "is the object being iterated. If "*
             "`IterativeControl.expose(m)` has been overloaded, then "*
             "log `f(expose(m))` instead.\n\n"*
             "Can be suppressed by setting the global verbosity level "*
             "sufficiently low. \n\n"*
             "See also [`Warn`](@ref), [`Error`](@ref). ")

function update!(c::Info, model, verbosity, args...)
    state = _log_eval(c.f, model)
    verbosity > 0 && @info state
    return state
end
public_state(::Info, state) = (; called_back=state)


# # Warn

struct Warn{P<:Function,F<:Union{Function,String}}
    predicate::P
    f::F
end

# constructor:
Warn(predicate; f="") = Warn(predicate, f)

@create_docs(Warn,
             header="Warn(predicate; f=\"\")",
             example="Warn(m -> length(m.cache) > 100, "*
             "f=\"Memory low\")",
             body="If `predicate(m)` is `true`, then "*
             "log to `Warn` the value of `f` "*
             "(or `f(IterationControl.expose(m))` if `f` is a function). "*
             "Here `m` "*
             "is the object being iterated.\n\n"*
             "Can be suppressed by setting the global verbosity level "*
             "sufficiently low.\n\n"*
             "See also [`Info`](@ref), [`Error`](@ref). ")

function update!(c::Warn, model, verbosity, n, state=(warnings=(),))
    warnings = state.warnings
    if c.predicate(model)
        warning = _log_eval(c.f, model)
        verbosity > -1 && @warn warning
        newstate = (warnings = tuple(warnings..., warning),)
    else
        newstate = state
    end
    return newstate
end
public_state(c::Warn, state) = (;)
_warnings(n) = n > 1 ? "warnings" : "warning"
function takedown(c::Warn, verbosity, state)
    n_warn = length(state.warnings)
    message = isempty(state.warnings) ? "" :
        "A `Warn` control issued $n_warn $(_warnings(n_warn))."
    verbosity > -1  && !isempty(message) && @warn message
    return message
end


# # Error

struct Error{P<:Function,F<:Union{Function,String}}
    predicate::P
    f::F
    exception::Union{Exception, Nothing}
end

# constructor:
Error(predicate; f="", exception=nothing) = Error(predicate, f, exception)

@create_docs(Error,
             header="Error(predicate; f=\"\", exception=nothing))",
             example="Error(m -> isnan(m.bias), f=\"Bias overflow!\")",
             body="If `predicate(m)` is `true`, then "*
             "log at the `Error` level the value of `f` "*
             "(or `f(IterationControl.expose(m))` if `f` is a function) "*
             "and stop iteration at the end of the current control cycle. "*
             "Here `m` "*
             "is the object being iterated.\n\n"*
             "Specify `exception=...` to throw an immediate "*
             "exception, without "*
             "waiting to the end of the control cycle.\n\n"*
             "See also [`Info`](@ref), [`Warn`](@ref). ")

function update!(
    c::Error,
    model,
    verbosity,
    n=1,
    state=(done = false, error=nothing),
    )
    if c.predicate(model)
        error = _log_eval(c.f, model)
        @error error
        c.exception === nothing || throw(c.exception)
        state = (done = true, error=error)
    end
    return state
end
public_state(::Error, state) = (; error=state.error)
done(c::Error, state) = state.done

takedown(c::Error, verbosity, state) = string(state.error)


# # CALLBACK

struct Callback{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
    raw::Bool
end

# constructor:
Callback(
    ; f=identity,
    stop_if_true=false,
    stop_message=nothing,
    raw=false,
) = Callback(f, stop_if_true, stop_message, raw)
Callback(f; kwargs...) = Callback(; f=f, kwargs...)

@create_docs(
    Callback,
    header="Callback(f=_->nothing, stop_if_true=false, "*
        "stop_message=nothing, raw=false)",
    example="Callback(m->put!(v, my_loss_function(m))",
    body="Call `f(IterationControl.expose(m))`, where "*
        "`m` is the object being iterated, unless `raw=true`, in "*
        "which case call `f(m)` (guaranteed if `expose` has not been "*
        "overloaded.) "*
        "If `stop_if_true` is `true`, then trigger an early stop "*
        "if the value returned by `f` is `true`, logging the "*
        "`stop_message` if specified. ",
)

update!(c::Callback, model, verbosity, n, state=nothing) = c.f(expose(model, c.raw))
public_state(::Callback, state) = (; called_back=state)
done(c::Callback, state) = c.stop_if_true && state isa Bool && state

function takedown(c::Callback, verbosity, state)
    if done(c, state)
        message = isnothing(c.stop_message) ?
            "Stop triggered by a `Callback` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return message
    else
        message = ""
    end
    return ""
end


# # DATA

struct Data{S}
    data::S
    stop_when_exhausted::Bool
end
Data(; data=(), stop_when_exhausted=false) = Data(data, stop_when_exhausted)
Data(data; kwargs...) = Data(data=data; kwargs...)

Base.show(io::IO, d::Data{S}) where S =
    print(io, "Data{$S}(<omitted data>; "*
          "stop_when_exhausted=$(d.stop_when_exhausted))")

@create_docs(Data,
             header="Data(my_data; stop_when_exhausted=false)",
             example="Data(rand(100))",
             body="In each application of this control a new `item` from the "*
             "iterable, `data`, is retrieved (using `iterate`) and "*
             "`IterationControl.ingest!(m, item)` is called. Here "*
             "`m` is the object being iteratively trained. \n\n"*
             "The control becomes passive once the `data` iterable is done. "*
             "To trigger "*
             "a stop *after one passive application of the control*, set "*
             "`stop_when_exhausted=true`. ")

const DATA_EXHAUSTED = "Data exhausted."
const DATA_STOP ="Stop triggered because data exhausted."

function update!(c::Data, model, verbosity, n)
    iter = c.data
    data_exhausted = true
    next = iter_state = iterate(iter)
    if next !== nothing
        data_exhausted = false
        item, iter_state = next
    end
    data_exhausted && verbosity > 0 && !c.stop_when_exhausted &&
        @info DATA_EXHAUSTED
    data_exhausted || ingest!(model, item)
    done = data_exhausted && c.stop_when_exhausted
    return (iter_state = iter_state, done = done)
end

function update!(c::Data, model, verbosity, n, state)
    iter = c.data
    iter_state = state.iter_state
    data_exhausted = true
    if iter_state !== nothing
        next = iterate(iter, iter_state)
        if next === nothing
            iter_state = nothing
        else
            data_exhausted = false
            item, iter_state = next
        end
    end
    data_exhausted && verbosity > 0 && !c.stop_when_exhausted &&
        iter_state !== nothing && @info DATA_EXHAUSTED
    data_exhausted || ingest!(model, item)
    done = data_exhausted && c.stop_when_exhausted
    return (iter_state = iter_state, done = done)
end

public_state(::Data, state) = (;)

done(c::Data, state) = state.done

function takedown(c::Data, verbosity, state)
    if state.done
        verbosity > 0 && @info DATA_STOP
        return DATA_STOP
    end
    return ""
end

needs_ingest(::Data) = true


# # WithLossDo

struct WithLossDo{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
WithLossDo(
    ; f=identity,
    stop_if_true=false,
    stop_message=nothing,
     ) = WithLossDo(f, stop_if_true, stop_message)
WithLossDo(f; kwargs...) = WithLossDo(; f=f, kwargs...)

@create_docs(WithLossDo,
             header="WithLossDo(f=identity, "*
             "stop_if_true=false, "*
             "stop_message=nothing)",
             example="WithLossDo(x->put!(my_losses, x))",
             body="Call `f(loss)`, where "*
             "`loss` is current loss.\n\n"*
             "If `stop_if_true` is `true`, then trigger an early stop "*
             "if the value returned by `f` is `true`, logging the "*
             "`stop_message` if specified. ")

needs_loss(::WithLossDo) = true

function update!(
    c::WithLossDo,
    model,
    verbosity,
    n,
    state=(; loss=nothing, called_back=nothing),
    )
    loss = IterationControl.loss(model)
    loss === nothing && throw(ERR_NEEDS_LOSS)
    called_back = c.f(loss)
    return (; loss, called_back)
end

function done(c::WithLossDo, state)
    called_back = state.called_back
    return c.stop_if_true && called_back isa Bool && called_back
end

function takedown(c::WithLossDo, verbosity, state)
    if done(c, state)
        message = c.stop_message === nothing ?
            "Stop triggered by a `WithLossDo` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return message
    end
    return ""
end


# # WithTrainingLossesDo

struct WithTrainingLossesDo{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
WithTrainingLossesDo(
    ; f=identity,
    stop_if_true=false,
    stop_message=nothing,
) = WithTrainingLossesDo(f, stop_if_true, stop_message)
WithTrainingLossesDo(f; kwargs...) =
    WithTrainingLossesDo(; f=f, kwargs...)

@create_docs(WithTrainingLossesDo,
             header="WithTrainingLossesDo(f=identity, "*
             "stop_if_true=false, "*
             "stop_message=nothing)",
             example="WithTrainingLossesDo(v->put!(my_losses, last(v))",
             body="Call `f(training_losses)`, where "*
             "`training_losses` is the vector of most recent batch "*
             "of training losses.\n\n"*
             "If `stop_if_true` is `true`, then trigger an early stop "*
             "if the value returned by `f` is `true`, logging the "*
             "`stop_message` if specified. ")

needs_training_losses(::WithTrainingLossesDo) = true

function update!(c::WithTrainingLossesDo,
                 model,
                 verbosity,
                 n,
                 state=(latest_training_loss = nothing, called_back=nothing))
    losses = IterationControl.training_losses(model)
    losses === nothing && throw(ERR_NEEDS_TRAINING_LOSSES)
    called_back = c.f(losses)
    return (; training_loss=losses[end], called_back)
end

function done(c::WithTrainingLossesDo, state)
    called_back = state.called_back
    return c.stop_if_true && called_back isa Bool && called_back
end

function takedown(c::WithTrainingLossesDo, verbosity, state)
    if done(c, state)
        message = c.stop_message === nothing ?
            "Stop triggered by a `WithTrainingLossesDo` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return message
    end
    return ""
end


# # WithNumberDo

struct WithNumberDo{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
WithNumberDo(
    ; f=identity,
    stop_if_true=false,
    stop_message=nothing,
) = WithNumberDo(f, stop_if_true, stop_message)
WithNumberDo(f; kwargs...) = WithNumberDo(; f=f, kwargs...)

@create_docs(WithNumberDo,
             header="WithNumberDo(f=identity, "*
             "stop_if_true=false, "*
             "stop_message=nothing)",
             example="WithNumberDo(n->put!(my_channel, n))",
             body="Call `f(n + 1)`, where "*
             "`n` is the number of already completed control cycles "*
             "(so, `n = 1, 2, 3, ...`, unless control is "*
             "wrapped in a [`IterationControl.skip`](@ref)).\n\n"*
             "If `stop_if_true` is `true`, then trigger an early stop "*
             "if the value returned by `f` is `true`, logging the "*
             "`stop_message` if specified. ")

function update!(
    c::WithNumberDo,
    model,
    verbosity,
    n,
    state=(number=n, called_back=nothing),
    )
    called_back = c.f(n)
    return (; number = n, called_back)
end

function done(c::WithNumberDo, state)
    called_back = state.called_back
    return c.stop_if_true && called_back isa Bool && called_back
end

function takedown(c::WithNumberDo, verbosity, state)
    if done(c, state)
        message = c.stop_message === nothing ?
            "Stop triggered by a `WithNumberDo` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return message
    end
    return ""
end
