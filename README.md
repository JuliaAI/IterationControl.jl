IterationControl.jl

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/ablaom/IterationControl.jl/workflows/CI/badge.svg)](https://github.com/ablaom/IterationControl.jl/actions)| [![codecov.io](http://codecov.io/github/ablaom/IterationControl.jl/coverage.svg?branch=master)](http://codecov.io/github/ablaom/IterationControl.jl?branch=master) |

A lightweight package for controlling iterative algorithms, with a
view to training and optimizing machine learning models.

Builds on
[EarlyStopping.jl](https://github.com/ablaom/EarlyStopping.jl) and
inspired by
[LearningStrategies.jl](https://github.com/JuliaML/LearningStrategies.jl). 


## Installation

```julia
using Pkg
Pkg.add("IterationControl")
```

## Basic idea

Suppose you have [some kind of object](/examples/square_rooter/),
`SquareRooter(x)`, for iteratively computing approximations to the
square root of `x`:

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
using IterationControl
IterationControl.train!(model::SquareRooter, n) =  train!(model, n) # lifting
```
By definitiion, the lifted `train!` has the same functionality as the original one:

```julia
model = SquareRooter(9)
IterationControl.train!(model, 2)

julia> model.root
3.4
```
But now we can also do this:

```julia
julia> IterationControl.train!(model, Step(2), NumberLimit(3), Info(m->m.root));
[ Info: 3.4
[ Info: 3.00009155413138
[ Info: 3.0
[ Info: Stop triggered by NumberLimit(3) stopping criterion.
```

Here each control is repeatedly applied in sequence until one of them
triggers a stop. The first control `Step(2)` says, "Train the model
two more iterations"; the second asks, "Have I been applied 3 times
yet?", signalling a stop (at the end of the current control cycle) if
so; and the third logs the value of the function `m -> m.root`,
evaluated on `model`, to `Info`. In this example only the second
control can terminate model iteration.

If `model` admits a method returning a loss (in this case the
difference between `x` and the square of `root`) then we can lift
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
                               Step(1),
                               Threshold(0.0001),
                               Callback(callback));
[ Info: Stop triggered by Threshold(0.0001) stopping criterion.

julia> losses
2-element Array{Float64,1}:
 0.002439396192741583
 3.716891878724482e-7
```

In many appliations to machine learning, "loss" will be an
out-of-sample loss, computed after some iterations. If `model`
additionally generates user-inspectable "training losses" (one per
iteration) then similarly lifting the appropriate access function to
`IterationControl.training_losses` enables Prechelt's
progress-modified generalization loss stopping criterion, `PQ` (see
Table 1 below).

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
[this](/examples/tree_booster/) tree boosting example shows.


## Online and incremental training

For online or incremental training, lift the method for ingesting data
into the model to `IterationControl.ingest!(model, datum)` and use the
control `Data(data)`. Here `data` is any iterator generating the
`datum` items to be ingested (one per application of the control). By
default, the `Data` control becomes passive after `data` is
exhausted. Do `?Data` for details. (See [Access to model through a
wrapper](#access-to-model-through-a-wrapper) below on dealing with any
model wrapping necessary to implement data ingestion.)

A simple particle tracking example is given
[here](/examples/particle/).


## Verbose logging

The `IterationControl.train!` method can be given the keyword argument
`verbosity=...`, defaulting to `1`. The larger `verbosity`, the noisier.


## Controls provided

Controls are repeatedly applied in sequence until a control triggers a
stop. Each control type has a detailed doc-string. Below is a short
summary, with some advanced options omitted. 

control                 | description                                                                             | enabled if these are overloaded   | can trigger a stop | notation in Prechelt
------------------------|-----------------------------------------------------------------------------------------|-----------------------------------|-------|---------------
`Step(n=1)`             | Train model for `n` iterations                                                          |`train!`                           | no    |
`Info(f=identity)`      | Log to `Info` the value of `f(model)`                                                   |`train!`                           | no    |
`Warn(predicate, f="")` | Log to `Warn` the value of `f` or `f(model)` if `predicate(model)` holds                |`train!`                           | no    |
`Error(predicate, f="")`| Log to `Error` the value of `f` or `f(model)` if `predicate(model)` holds and then stop |`train!`                           | yes   |
`Callback(f=_->nothing)`| Call `f(model)`                                                                         |`train!`                           | yes   |
`TimeLimit(t=0.5)`      | Stop after `t` hours                                                                    |`train!`                           | yes   |
`NumberLimit(n=100)`    | Stop after `n` applications of the control                                              |`train!`                           | yes   |
`NumberSinceBest(n=6)`  | Stop when best loss occurred `n` control applications ago                               |`train!`                           | yes   |
`WithNumberDo(f=n->@info(n))`    | Call `f(n + 1)` where `n` is number of previous applications of control        |`train!`                           | yes   |
`WithLossDo(f=x->@info(x))`   | Call `f(loss)` where `loss` is the current loss                                   |`train!`, `loss`                   | yes   |
`WithTrainingLossesDo(f=v->@info(v))`| Call `f(v)` where `v` is the current batch of training losses              |`train!`, `training_loss`          | yes   |
`NotANumber()`          | Stop when `NaN` encountered                                                             |`train!`, `loss`                   | yes   |
`Threshold(value=0.0)`  | Stop when `loss < value`                                                                |`train!`, `loss`                   | yes   |
`GL(alpha=2.0)`         | Stop after "Generalization Loss" exceeds `alpha`                                        |`train!`, `loss`                   | yes   | ``GL_α``
`Patience(n=5)`         | Stop after `n` consecutive loss increases                                               |`train!`, `loss`                   | yes   | ``UP_s``
`PQ(alpha=0.75, k=5)`   | Stop after "Progress-modified GL" exceeds `alpha`                                       |`train!`, `loss`, `training_losses`| yes   | ``PQ_α``
`Data(data)`            | Call `ingest!(model, item)` on the next `item` in the iterable `data`.                  |`train!`, `ingest!`                | yes   |

> Table 1. Atomic controls

**Stopping option.** All the following controls trigger a stop if the
provided function `f` returns `true` and `stop_if_true=true` is
specified in the constructor: `Callback`, `WithNumberDo`, `WithLossDo`,
`WithTrainingLossesDo`.

There are also three control wrappers to modify a control's behavior:

wrapper                                            | description
---------------------------------------------------|-------------------------------------------------------------------------
`IterationControl.skip(control, predicate=1)`      | Apply `control` every `predicate` applications of the control wrapper (can also be a function; see doc-string)
`IterationControl.debug(control)`                  | Apply `control` but also log its state to `Info` (at any `verbosity` level)
`IterationControl.composite(controls...)`          | Apply each `control` in `controls` in sequence; mostly for under-the-hood use

> Table 2. Wrapped controls




## Access to model through a wrapper

Note that functions ordinarily applied to `model` by some control
(e.g., a `Callback`) will instead be applied to
`IterationControl.expose(model)` if `IterationControl.expose` is
appropriately overloaded.


## Implementing new controls

There is no abstract control type; any object can be a
control. Behaviour is implemented using a functional style interface
with four methods. Only the first two are compulsory (the `done` and
`takedown` fallbacks always return `false` and `NamedTuple()`
respectively.):

```julia
update!(control, model, verbosity) -> state  # initialization
update!(control, model, verbosity, state) -> state
done(control, state)::Bool
takedown(control, verbosity, state) -> human_readable_named_tuple
```

Here's how `IterationControl.train!` calls these methods:

```julia
function train!(model, controls...; verbosity::Int=1)

    control = composite(controls...)

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
```
