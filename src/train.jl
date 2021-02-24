function train!(model, controls...; verbosity::Int=1)

    control = CompositeControl(controls...)

    # before training:
    verbosity > 1 && @info "Using these controls: $(flat(control)). "

    # first training event:
    state = update!(control,
                    model,
                    verbosity - 1)
    finished = done(control, state)

    # subsequent training events:
    while !finished
        state = update!(control,
                        model,
                        verbosity - 1,
                        state)
        finished = done(control, state)
    end

    # finalization:
    return takedown(control, verbosity, state)
end
