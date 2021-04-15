# # TRAIN

struct Step
    n::Int
end

# constructor:
Step(; n=5) = Step(n)

@create_docs(Step,
             header="Step(; n=1)",
             example="Step(2)",
             body="Train for `n` more iterations. "*
             "Will never trigger a stop. ")

function update!(c::Step, model, verbosity, state=(n_iterations = 0,))
    n_iterations = state.n_iterations
    verbosity > 1 &&
        @info "Stepping model for $(c.n) more iterations. "
    train!(model, c.n)
    state  = (n_iterations = n_iterations + c.n,)
    return state
end

takedown(c::Step, verbosity, state) = state

# # Info

struct Info{F<:Function}
    f::F
end

# constructor:
Info(; f::Function=identity) = Info(f)

@create_docs(Info,
             header="Info(f=identity)",
             example="Info(my_loss_function)",
             body="Log at the `Info` level the value of `f(m)`, "*
             "where `m` "*
             "is the object being iterated. If "*
             "`IterativeControl.expose(m)` has been overloaded, then "*
             "log `f(expose(m))` instead.\n\n"*
             "Can be suppressed by setting the global verbosity level "*
             "sufficiently low. \n\n"*
             "See also [`Warn`](@ref), [`Error`](@ref). ")

function update!(c::Info, model, verbosity, args...)
    verbosity < 1 || @info _log_eval(c.f, model)
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
             example="Warn(m -> length(m.cache) > 100, "*
             "f=\"Memory low\")",
             body="If `predicate(m)` is `true`, then "*
             "log at the `Warn` level the value of `f` "*
             "(or `f(IterationControl.expose(m))` if `f` is a function). "*
             "Here `m` "*
             "is the object being iterated.\n\n"*
             "Can be suppressed by setting the global verbosity level "*
             "sufficiently low.\n\n"*
             "See also [`Info`](@ref), [`Error`](@ref). ")

function update!(c::Warn, model, verbosity, args...)
    verbosity > 1 && c.predicate(model) &&
        @warn _log_eval(c.f, model)
    return nothing
end

function update!(c::Warn, model, verbosity, warnings=())
    if c.predicate(model)
        warning = _log_eval(c.f, model)
        verbosity < 1 || @warn warning
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
             example="Error(m -> isnan(m.bias), f=\"Bias overflow!\")",
             body="If `predicate(m)` is `true`, then "*
             "log at the `Error` level the value of `f` "*
             "(or `f(IterationControl.expose(m))` if `f` is a function) "*
             "and stop iteration at the end of the current control cycle. "*
             "Here `m` "*
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
             body="Call `f(IterationControl.expose(m))`, where "*
             "`m` is the object being iterated, unless `raw=true`, in "*
             "which case call `f(m)` (guaranteed if `expose` has not been "*
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
            "Stop triggered by a `Callback` control. " :
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
             header="Data(my_data; stop_when_exhausted=false)",
             example="Data(rand(100))",
             body="In each application of this control a new `item` from the "*
             "iterable, `data`, is retrieved (using `iterate`) and "*
             "`IterationControl.ingest!(m, item)` is called. Here "*
             "`m` is the object being iterated. \n\n"*
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
    data_exhausted && verbosity > 0 && !c.stop_when_exhausted &&
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
    data_exhausted && verbosity > 0 && !c.stop_when_exhausted &&
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


# # WithLossDo

struct WithLossDo{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
WithLossDo(f::Function;
     stop_if_true=false,
     stop_message=nothing) = WithLossDo(f, stop_if_true, stop_message)
WithLossDo(; f=x->@info("loss: $x"), kwargs...) = WithLossDo(f, kwargs...)

@create_docs(WithLossDo,
             header="WithLossDo(f=x->@info(\"loss: \$x\"), "*
             "stop_if_true=false, "*
             "stop_message=nothing)",
             example="WithLossDo(x->put!(my_losses, x))",
             body="Call `f(loss)`, where "*
             "`loss` is current loss.\n\n"*
             "If `stop_if_true` is `true`, then trigger an early stop "*
             "if the value returned by `f` is `true`, logging the "*
             "`stop_message` if specified. ")

EarlyStopping.needs_loss(::Type{<:WithLossDo}) = true

function update!(c::WithLossDo,
                 model,
                 verbosity,
                 state=(loss=nothing, done=false))
    loss = IterationControl.loss(model)
    r = c.f(loss)
    done = (c.stop_if_true && r isa Bool && r) ? true : false
    return (loss=loss, done=done)
end

done(c::WithLossDo, state) = state.done

function takedown(c::WithLossDo, verbosity, state)
    if state.done
        message = c.stop_message === nothing ?
            "Stop triggered by a `WithLossDo` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return merge(state, (log = message,))
    else
        return merge(state, (log = "",))
    end
end


# # WithTrainingLossesDo

struct WithTrainingLossesDo{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
WithTrainingLossesDo(f::Function;
     stop_if_true=false,
     stop_message=nothing) = WithTrainingLossesDo(f, stop_if_true, stop_message)
WithTrainingLossesDo(; f=v->@info("training: $v"), kwargs...) =
    WithTrainingLossesDo(f, kwargs...)

@create_docs(WithTrainingLossesDo,
             header="WithTrainingLossesDo(f=v->@info(\"training: \$v\"), "*
             "stop_if_true=false, "*
             "stop_message=nothing)",
             example="WithTrainingLossesDo(v->put!(my_losses, last(v))",
             body="Call `f(training_losses)`, where "*
             "`training_losses` is the vector of most recent batch "*
             "of training losses.\n\n"*
             "If `stop_if_true` is `true`, then trigger an early stop "*
             "if the value returned by `f` is `true`, logging the "*
             "`stop_message` if specified. ")

EarlyStopping.needs_training_losses(::Type{<:WithTrainingLossesDo}) = true

function update!(c::WithTrainingLossesDo,
                 model,
                 verbosity,
                 state=(latest_training_loss = nothing, done = false))
    losses = IterationControl.training_losses(model)
    r = c.f(losses)
    done = (c.stop_if_true && r isa Bool && r) ? true : false
    return (latest_training_loss=losses[end], done=done)
end

done(c::WithTrainingLossesDo, state) = state.done

function takedown(c::WithTrainingLossesDo, verbosity, state)
    if state.done
        message = c.stop_message === nothing ?
            "Stop triggered by a `WithTrainingLossesDo` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return merge(state, (log = message,))
    else
        return merge(state, (log = "",))
    end
end


# # WithNumberDo

struct WithNumberDo{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
WithNumberDo(f::Function;
     stop_if_true=false,
     stop_message=nothing) = WithNumberDo(f, stop_if_true, stop_message)
WithNumberDo(; f=n->@info("number: $n"), kwargs...) = WithNumberDo(f, kwargs...)

@create_docs(WithNumberDo,
             header="WithNumberDo(f=n->@info(\"number: \$n\"), "*
             "stop_if_true=false, "*
             "stop_message=nothing)",
             example="WithNumberDo(n->put!(my_channel, n))",
             body="Call `f(n)`, where "*
             "`n` is one more than the number of previous applications "*
             "of the control (so, `n = 1, 2, 3, ...`).\n\n"*
             "If `stop_if_true` is `true`, then trigger an early stop "*
             "if the value returned by `f` is `true`, logging the "*
             "`stop_message` if specified. ")

function update!(c::WithNumberDo, model, verbosity, state=(done = false, n = 0))
    n = state.n
    r = c.f(state.n + 1)
    done = (c.stop_if_true && r isa Bool && r) ? true : false
    return (done = done, n = n + 1)
end

done(c::WithNumberDo, state) = state.done

function takedown(c::WithNumberDo, verbosity, state)
    if state.done
        message = c.stop_message === nothing ?
            "Stop triggered by a `WithNumberDo` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return merge(state, (log = message,))
    else
        return merge(state, (log = "",))
    end
end
