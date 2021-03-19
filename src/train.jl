const ERR_TRAIN = ArgumentError("`IterationControl.train!` needs at "*
    "least two arguments. ")

function train!(model, controls...; verbosity::Int=1)

    isempty(controls) && throw(ERR_TRAIN)

    control = CompositeControl(controls...)

    # before training:
    verbosity > 1 && @info "Using these controls: $(flat(control)). "

    # first training event:
    state = update!(control, model, verbosity)
    finished = done(control, state)

    # subsequent training events:
    while !finished
        state = update!(control, model, verbosity, state)
        finished = done(control, state)
    end

    # finalization:
    return takedown(control, verbosity, state)
end
