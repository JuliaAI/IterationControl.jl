# Adding New Controls

## Implementing new controls

There is no abstract control type; any object can be a control. Behavior is implemented
using a functional style interface documented below. Only the first two methods below are
compulsory.

Main methods:

- [`IterationControl.update!(control, model, verbosity, n)`](@ref)` -> state`  # initialization
- [`IterationControl.update!(control, model, verbosity, n, state)`](@ref)` -> state`
- [`EarlyStopping.public_state(control, state)`](@ref)
- [`IterationControl.done(control, state)`](@ref)`::Bool`
- [`IterationControl.takedown(control, verbosity, state)`](@ref)` -> message`

Here `n` is the control cycle count, i.e., one more than the the
number of completed control cycles.

Control traits (fallback to `false`):

- [`IterationControl.needs_loss(control)`](@ref)
- [`IterationControl.needs_training_losses(control)`](@ref)
- [`IterationControl.needs_ingest(control)`](@ref)


Simplified version of how `IterationControl.train!` calls these methods:

```julia
function train!(model, controls...; verbosity::Int=1)

    control = composite(controls...)

    # first training event:
    n = 1 # counts control cycles
    state = update!(control, model, verbosity, n)
    finished = done(control, state)

    <code that initializes the control traces>
    <and writes to each trace the current public state>

    # subsequent training events:
    while !finished
        n += 1
        state = update!(control, model, verbosity, n, state)
        finished = done(control, state)
        <code to update the traces with current public state>
    end

    <call `takedown(control, verbosity, state)` on each component control>
    <and store the the returned messages to the traces>

    <return the control traces>
end
```

```@docs
IterationControl.update!
EarlyStopping.public_state
IterationControl.done
IterationControl.takedown
IterationControl.needs_loss
IterationControl.needs_training_losses
IterationControl.needs_ingest
```
