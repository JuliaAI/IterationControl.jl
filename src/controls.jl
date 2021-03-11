# # TRAIN

struct Step
    n::Int
end

# constructor:
Step(; n=5) = Step(n)

@create_docs(Step,
             header="Step(; n=1)",
             example="Step(2)",
             body="Train the model for `n` more iterations. "*
             "Will never trigger a stop. ")

function update!(c::Step, model, verbosity, args...)
    if verbosity > 0
        @info "Steping model for $(c.n) iterations. "
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
             example="Info(my_loss_function)",
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
             header="Warn(predicate; f=\"\")",
             example="Warn(model -> length(model.cache) > 100, "*
             "f=\"Memory low\")",
             body="If `predicate(model)` is `true`, then "*
             "log at the `Warn` level the value of `f` "*
             "(or `f(IterationControl.expose(model))` if `f` is a function). "*
             "Here `model` "*
             "is the object being iterated.\n\n"*
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
             header="Error(predicate; f=\"\", exception=nothing))",
             example="Error(model -> isnan(model.bias), f=\"Bias overflow!\")",
             body="If `predicate(model)` is `true`, then "*
             "log at the `Error` level the value of `f` "*
             "(or `f(IterationControl.expose(model))` if `f` is a function) "*
             "and stop iteration at the end of the current control cycle. "*
             "Here `model` "*
             "is the object being iterated.\n\n"*
             "Specify `exception=...` to throw an immediate "*
             "execption, without "*
             "waiting to the end of the control cycle.\n\n"*
             "See also [`Info`](@ref), [`Warn`](@ref). ")

function update!(c::Error,
                 model,
                 verbosity,
                 state=(done = false, error=()))
    if c.predicate(model)
        error = _log_eval(c.f, model)
        @error error
        c.exception === nothing || throw(c.exception)
        state = (done = true, error=error)
    end
    return state
end

done(c::Error, state) = state.done

takedown(c::Error, verbosity, state) = state


# # CALLBACK

struct Callback{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
    raw::Bool
end

# constructor:
Callback(f::Function;
         stop_if_true=false,
         stop_message=nothing,
         raw=false) = Callback(f, stop_if_true, stop_message, raw)
Callback(; f=identity, kwargs...) = Callback(f, kwargs...)

@create_docs(Callback,
             header="Callback(f=_->nothing, stop_if_true=false, "*
             "stop_message=nothing, raw=false)",
             example="Callback(m->put!(v, my_loss_function(m))",
             body="Call `f(IterationControl.expose(model))`, where "*
             "`model` is the object being iterated, unless `raw=true`, in "*
             "which case call `f(model)` (guaranteed if `expose` has not been "*
             "overloaded.) "*
             "If `stop_if_true` is `true`, then trigger an early stop "*
             "if the value returned by `f` is `true`, logging the "*
             "`stop_message` if specified. ")

function update!(c::Callback, model, verbosity, state=(done=false, ))
    r = c.f(expose(model, c.raw))
    done = (c.stop_if_true && r isa Bool && r) ? true : false
    return (done=done,)
end

done(c::Callback, state) = state.done

function takedown(c::Callback, verbosity, state)
    if state.done
        message = c.stop_message === nothing ?
            "Early stop triggered by a `Callback` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return (done = true, log = message)
    else
        return (done = false, log = "")
    end
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
             header="Data(data; stop_when_exhausted=false)",
             example="Data(rand(100))",
             body="In each application of this control a new `item` from the "*
             "iterable, `data`, is retrieved (using `iterate`) and "*
             "`IterationControl.ingest!(model, item)` is called. Here "*
             "`model` is the object being iterated. \n\n"*
             "A control becomes passive once the `data` iterable is done. "*
             "To trigger "*
             "a stop *after one passive application of the control*, set "*
             "`stop_when_exhausted=true`. ")

const DATA_EXHAUSTED = "Data exhausted. "
const DATA_STOP ="Stop triggered because data exhausted ."

function update!(c::Data, model, verbosity)
    iter = c.data
    data_exhausted = true
    next = iter_state = iterate(iter)
    if next !== nothing
        data_exhausted = false
        item, iter_state = next
    end
    data_exhausted && verbosity > -1 && !c.stop_when_exhausted &&
        @info DATA_EXHAUSTED
    data_exhausted || ingest!(model, item)
    done = data_exhausted && c.stop_when_exhausted
    return (iter_state = iter_state, done = done)
end

function update!(c::Data, model, verbosity, state)
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
    data_exhausted && verbosity > -1 && !c.stop_when_exhausted &&
        iter_state !== nothing && @info DATA_EXHAUSTED
    data_exhausted || ingest!(model, item)
    done = data_exhausted && c.stop_when_exhausted
    return (iter_state = iter_state, done = done)
end

done(c::Data, state) = state.done

function takedown(c::Data, verbosity, state)
    if state.done
        verbosity > 0 && @info DATA_STOP
        return (done = true, log = DATA_STOP)
    else
        return (done = false, log = "")
    end
end


# # Loss

struct Loss{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
Loss(f::Function;
     stop_if_true=false,
     stop_message=nothing) = Loss(f, stop_if_true, stop_message)
Loss(; f=x->@info(x), kwargs...) = Loss(f, kwargs...)

@create_docs(Loss,
             header="Loss(f=x->@info(x)), stop_if_true=false, "*
             "stop_message=nothing)",
             example="Loss(x->put!(my_losses, x)",
             body="Call `f(loss)`, where "*
             "`loss` is current loss.\n\n"*
             "If `stop_if_true` is `true`, then trigger an early stop "*
             "if the value returned by `f` is `true`, logging the "*
             "`stop_message` if specified. ")

EarlyStopping.needs_loss(::Type{<:Loss}) = true

function update!(c::Loss, model, verbosity, state=(done=false, ))
    _loss = get_loss(c, model)
    r = c.f(_loss)
    done = (c.stop_if_true && r isa Bool && r) ? true : false
    return (done=done,)
end

done(c::Loss, state) = state.done

function takedown(c::Loss, verbosity, state)
    if state.done
        message = c.stop_message === nothing ?
            "Early stop triggered by a `Loss` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return (done = true, log = message)
    else
        return (done = false, log = "")
    end
end
