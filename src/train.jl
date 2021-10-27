const ERR_TRAIN = ArgumentError("`IterationControl.train!` needs at "*
    "least two arguments. ")

function train!(model, controls...; verbosity::Int=1)

    isempty(controls) && throw(ERR_TRAIN)

    control = CompositeControl(controls...)

    # before training:
    verbosity > 1 && @info "Using these controls: $(flat(control)). "

    # first training event:
    n = 1 # counts control cycles
    state = update!(control, model, verbosity, n)
    finished = done(control, state)

    # subsequent training events:
    while !finished
        n += 1
        state = update!(control, model, verbosity, n, state)
        finished = done(control, state)
    end

    # reporting final loss and training loss if available:
    loss = IterationControl.loss(model)
    training_losses = IterationControl.training_losses(model)
    if verbosity > 0
        loss isa Nothing || @info "final loss: $loss"
        training_losses isa Nothing || isempty(training_losses) ||
            @info "final training loss: $(training_losses[end])"
        verbosity > 1 && @info "total control cycles: $n"
    end

    # finalization:
    return takedown(control, verbosity, state)
end
