const ERR_TRAIN = ArgumentError("`IterationControl.train!` needs at "*
                                "least two arguments. ")

const ERR_NEEDS_LOSS = ArgumentError(
    "Encountered a control that needs losses but no losses found. ")

const ERR_NEEDS_TRAINING_LOSSES = ArgumentError(
    "Encountered a control that needs training losses but no training "*
    "losses found. ")

function train!(model, controls...; verbosity::Int=1)

    isempty(controls) && throw(ERR_TRAIN)

    control = composite(controls...)

    # before training:
    verbosity > 1 && @info "Using these controls: $(flat(control)). "

    # first training event:
    n = 1 # counts control cycles
    state = update!(control, model, verbosity, n)
    finished = done(control, state)

    # checks that model supports control:
    if needs_loss(control) && loss(model) === nothing
        throw(ERR_NEEDS_LOSS)
    end
    if needs_training_losses(control) && training_losses(model) === nothing
        throw(ERR_NEEDS_TRAINING_LOSSES)
    end

    # subsequent training events:
    while !finished
        n += 1
        state = update!(control, model, verbosity, n, state)
        finished = done(control, state)
    end

    # reporting final loss and training loss if available:
    _loss = IterationControl.loss(model)
    _training_losses = IterationControl.training_losses(model)
    if verbosity > 0
        _loss isa Nothing || @info "final loss: $_loss"
        _training_losses isa Nothing || isempty(_training_losses) ||
            @info "final training loss: $(_training_losses[end])"
        verbosity > 1 && @info "total control cycles: $n"
    end

    # finalization:
    return takedown(control, verbosity, state)
end
