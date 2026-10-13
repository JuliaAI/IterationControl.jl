const ERR_TRAIN = ArgumentError("`IterationControl.train!` needs at "*
                                "least two arguments. ")

const ERR_NEEDS_LOSS = ArgumentError(
    "Encountered a control that needs losses but no losses found. ")

const ERR_NEEDS_TRAINING_LOSSES = ArgumentError(
    "Encountered a control that needs training losses but no training "*
    "losses found. ")

const ERR_TRACES_NEED_INITIALIZING = ArgumentError(
    "The keyword argument `traces` must point to an *empty* vector `[]`. "*
        "Perhaps you populated `traces` with a previous `train!` run. "
)

"""
    train!(model, controls...; verbosity=1, compact=false, traces=[])

Train the iterative `model`, applying the provided `controls`, which must include the
`Step(k)` control, for some integer `k` that specifies the number of training iterations
between control cycles.

Returns a vector of traces, one for each control. A trace records
the evolving internal state of each control, unless `compact=false`, in which case only
the current state is stored.

```julia-repl
using IterationControl
model = IterationControl.SquareRooter(4)
controls = [Step(1), Threshold(1e-7), NumberLimit(100), InvalidValue()]

julia> traces = IterationControl.train!(model, controls...)
[ Info: final loss: 8.881784197001252e-15
[ Info: final training loss: 9.292229252721995e-8
[ Info: Stop triggered by Threshold(1.0e-7) stopping criterion.

julia> model.root
2.000000000000002

julia> traces[2]
ControlTrace
  control: Threshold(1.0e-7)
  loss: [ …, 0.002439396192741583, 3.716891878724482e-7, 8.881784197001252e-15]
  excess: [ …, 0.002439296192741583, 2.716891878724482e-7, -9.99999911182158e-8]
  table: (loss = …, excess = …)
  message: Stop triggered by Threshold(1.0e-7) stopping criterion.
  triggered_stop: true

julia> traces[2].loss
5-element Vector{Float64}:
 2.25
 0.20249999999999968
 0.002439396192741583
 3.716891878724482e-7
 8.881784197001252e-15
```

It is assumed that `IterationControl.train!(model, n::Int)` has been implemented, and that
this trains the model for `n` iterations. Specific controls may require implementations of
`IterationControl.loss`, `IterationControl.training_losses` and/or
`IterationControl.ingest!`. Refer to documentation for details.

# Extended help

# Control traces and live tracking

For "live" access to the traces, while `train!` is running as an
asynchronous task, specify `traces=v`, where `v` is bound to some copy of `Any[]` to be
populated by the traces once the first control cycle completes, for subsequent mutation as
iteration proceeds:

```julia-repl
model = IterationControl.SquareRooter(4)
controls = [Step(1), Threshold(1e-7), Callback(_->sleep(0.5))]
traces=[]
Threads.@spawn IterationControl.train!(model, controls...; verbosity=0, traces=traces)
length(traces) # 0
sleep(3)
length(traces) # 3

julia> traces
3-element Vector{Any}:
 ControlTrace{<:Step}, ending (new_iterations = 5, new_steps = 5)
 ControlTrace{<:Threshold}, ending (loss = 8.881784197001252e-15, excess = -9.99999911182158e-8)
 ControlTrace{<:Callback}, ending (called_back = nothing,)
```

"""
function train!(model, controls...; verbosity::Int=1, compact=false, traces=[])

    isempty(traces) || throw(ERR_TRACES_NEED_INITIALIZING)
    isempty(controls) && throw(ERR_TRAIN)

    control = composite(controls...)

    # before training:
    verbosity > 1 && @info "Using these controls: $(flat(control)). "

    # first training event:
    n = 1 # counts control cycles
    state = update!(control, model, verbosity, n)
    append!(traces, broadcast(ControlTrace, flat(control), flat(state)))
    finished = done(control, state)

    # checks that model supports control:
    if needs_loss(control) && isnothing(loss(model))
        throw(ERR_NEEDS_LOSS)
    end
    if needs_training_losses(control) && isnothing(training_losses(model))
        throw(ERR_NEEDS_TRAINING_LOSSES)
    end

    # subsequent training events:
    while !finished
        n += 1
        state = update!(control, model, verbosity, n, state)
        if compact
            broadcast(update!, traces, flat(state))
        else
            broadcast(push!, traces, flat(state))
        end
        finished = done(control, state)
    end

    # reporting final loss and training loss if available:
    _loss = IterationControl.loss(model)
    _training_losses = IterationControl.training_losses(model)
    if verbosity > 0
        _loss isa Nothing || @info "final loss: $_loss"
        _training_losses isa Nothing || isempty(_training_losses) ||
            @info "final training loss: $(_training_losses[end])"
        verbosity > 1 && @info "total control cycles (steps): $n"
    end

    # finalization:
    broadcast(takedown!, traces, Ref(verbosity), flat(state))
    return collect(traces)
end
