# IterationControl.jl

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/ablaom/IterationControl.jl/workflows/CI/badge.svg)](https://github.com/ablaom/IterationControl.jl/actions)| [![codecov.io](http://codecov.io/github/ablaom/IterationControl.jl/coverage.svg?branch=master)](http://codecov.io/github/ablaom/IterationControl.jl?branch=master) |

A package for controlling iterative algorithms.

Not registered and still experimental.

To do: Add interface point for online learning


## Installation

```julia
using Pkg
Pkg.add("IterativeControl")
```

## Basic idea

Suppose you have some kind of object `SquareRooter(x)` for
iteratively computing approximations to the square roots of `x`:

```julia
model = SquareRooter(9)

julia> model.root
1.0

train!(model, 2) # train for 2 iterations

julia> model.root
3.4

train!(model, 1) # train for 1 more iteration

julia> model.root
3.023529411764706
```

Then we can replace the integer argument `n` in `train!(model, n)`
with a number of more sophisticated *controls* by "lifting" the method
`train!` to the `IterationControl.train!` method defined in this
package:

```julia
using IterativeControl
IterationControl.train!(model::SquareRooter, n) =  train!(model, n) # lifting
```
The lifted `train!` has the same functionality as the original one:

```julia
model = SquareRooter(9)
IterationControl.train!(model, 2)

julia> model.root
3.4
```
But now we can also do this:

```julia
julia> IterationControl.train!(model, Train(2), NumberLimit(3), Info(m->m.root));
[ Info: 3.4
[ Info: 3.00009155413138
[ Info: 3.0
[ Info: Early stop triggered by NumberLimit(3) stopping criterion.
```

Here each control is repeatedly applied until one of them triggers a
stop. The first control `Train(2)` says "train the model two more
iterations"; the second says "stop after 3 repetitions" (of the
sequence of control applications); and the third, "log the value of
the root to `Info`".

If `model` admits a method returning a loss (for example, the
difference between `x` and the square of `root`), then we can lift
that method to `IterationControl.loss` to enable control using
loss-based stopping criteria, such as a loss threshold. In the
demonstation below, we also include a callback:

```julia
model = SquareRooter(4)
train!(model, 1)

julia> loss(model)
2.25

IterationControl.loss(model::SquareRooter) = loss(model) # lifting

losses = Float64[]
callback(model) = push!(losses, loss(model))

julia> IterationControl.train!(model,
                               Train(1),
                               Threshold(0.0001),
                               Callback(callback));
[ Info: Early stop triggered by Threshold(0.0001) stopping criterion.

julia> losses
2-element Array{Float64,1}:
 0.002439396192741583
 3.716891878724482e-7
```

If training `model` generates user-inspectable "training losses" (one
per iteration) then similarly lifting the appropriate access function
to `IterationControl.training_losses` enables Prechelt's
progress-modified generalization loss stopping criterion, `PQ`.

`PQ` is the only criterion from the
[EarlyStopping.jl](https://github.com/ablaom/EarlyStopping.jl) package
not otherwise enabled when `IterationControl.loss` is overloaded as
above.

*Reference.* [Prechelt, Lutz
 (1998)](https://link.springer.com/chapter/10.1007%2F3-540-49430-8_3):
 "Early Stopping - But When?", in *Neural Networks: Tricks of the
 Trade*, ed. G. Orr, Springer.

The interface just described is sufficient for controlling
conventional machine learning models with an iteration parameter, as
this [tree boosting example](/examples/iris/) shows. An extension of
the interface to handle online learning is planned.


## Controls provided

Controls are repeatedly applied in sequence until a control triggers
a stop. The first control in a sequence is generally
`Train(...)`. Each control type has a detailed doc-string. Here is
short summary, with some advanced options omitted:

control                 | description                                                                             | enabled if these are overloaded   | notation in Prechelt
------------------------|-----------------------------------------------------------------------------------------|-----------------------------------|----------------------
`Train(n=1)`            | Train model for `n` iterations                                                          |`train!`                           |
`Info(f=identity)`      | Log to `Info` the value of `f(model)`                                                   |`train!`                           |
`Warn(predicate, f="")` | Log to `Warn` the value of `f` or `f(model)` if `predicate(model)` holds                |`train!`                           |
`Error(predicate, f="")`| Log to `Error` the value of `f` or `f(model)` if `predicate(model)` holds and then stop |`train!`                           |
`Callback(f=_->nothing)`| Call `f(model)`                                                                         |`train!`                           |
`TimeLimit(t=0.5)`      | Stop after `t` hours                                                                    |`train!`                           |
`NumberLimit(n=100)`    | Stop after `n` loss updates (excl. "training losses")                                   |`train!`                           |
`NotANumber()`          | Stop when `NaN` encountered                                                             |`train!`, `loss`                   |
`Threshold(value=0.0)`  | Stop when `loss < value`                                                                |`train!`, `loss`                   |
`GL(alpha=2.0)`         | Stop after "Generalization Loss" exceeds `alpha`                                        |`train!`, `loss`                   | ``GL_α``
`Patience(n=5)`         | Stop after `n` consecutive loss increases                                               |`train!`, `loss`                   | ``UP_s``
`PQ(alpha=0.75, k=5)`   | Stop after "Progress-modified GL" exceeds `alpha`                                       |`train!`, `loss`, `training_losses`| ``PQ_α``


> Table 1. Atomic controls

There are also three control wrappers to modify a control's behavior:

wrapper                                            | description
---------------------------------------------------|-------------------------------------------------------------------------
`IterationControl.skip(control, predicate=1)`      | Apply `control` every `predicate` applications of the control wrapper (can also be a function; see doc-string)
`IterationControl.debug(control)`                  | Apply `control` but also log its state to `Info` (at any `verbosity` level)
`IterationControl.composite(controls...)`          | Apply each `control` in `controls` in sequence; mostly for under-the-hood use

> Table 2. Wrapped controls


## Verbose logging

The `IterationControl.train!` method can be given the keyword argument
`verbosity=...`, defaulting to `1`. The larger `verbosity`, the noisier.


## Access to model through a wrapper

Note that predicates ordinarily applied to `model` by some control
(e.g., a `Callback`) will instead be applied to
`IterationControl.expose(model)` if `IterationControl.expose` is
appropriately overloaded.


## Implementing new controls

There is no abstract control type; any object can be a
control. Behaviour is implemented using a functional style interface
with four methods. Only the first two are compulsory:

```julia
update!(control, model, verbosity) -> state  # initialization
update!(control, model, verbosity, state) -> state
done(control, state)::Bool
takedown(control, verbosity, state) -> human_readable_named_tuple
```

Here's how `IterationControl.train!` calls these methods:

```julia
function train!(model, controls...; verbosity::Int=1)

    control = CompositeControl(controls...)

    # before training:
    verbosity > 1 && @info "Using these controls: $(flat(control)). "

    # first training event:
    state = update!(control, model, verbosity - 1)
    finished = done(control, state)

    # subsequent training events:
    while !finished
        state = update!(control, model, verbosity - 1, state)
        finished = done(control, state)
    end

    # finalization:
    return takedown(control, verbosity, state)
end
```
