# IterationControl.jl

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/JuliaAI/IterationControl.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/IterationControl.jl/actions)| [![codecov.io](http://codecov.io/github/JuliaAI/IterationControl.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaAI/IterationControl.jl?branch=master) |

A lightweight package for controlling iterative algorithms, with a
view to training and optimizing machine learning models.

Builds on
[EarlyStopping.jl](https://github.com/ablaom/EarlyStopping.jl) and
inspired by
[LearningStrategies.jl](https://github.com/JuliaML/LearningStrategies.jl).

Other related software:
[DynamicIterators.jl](https://github.com/mschauer/DynamicIterators.jl).

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
By definition, the lifted `train!` has the same functionality as the original one:

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
yet?", signaling a stop (at the end of the current control cycle) if
so; and the third logs the value of the function `m -> m.root`,
evaluated on `model`, to `Info`. In this example only the second
control can terminate model iteration.

If `model` admits a method returning a loss (in this case the
difference between `x` and the square of `root`) then we can lift
that method to `IterationControl.loss` to enable control using
loss-based stopping criteria, such as a loss threshold. In the
demonstration below, we also include a callback:

```julia
model = SquareRooter(4)
train!(model, 1)

julia> loss(model)
2.25

IterationControl.loss(model::SquareRooter) = loss(model) # lifting

losses = Float64[]
callback(model) = push!(losses, loss(model))

julia> IterationControl.train!(model, Step(1), Threshold(0.0001), Callback(callback));
[ Info: Stop triggered by Threshold(0.0001) stopping criterion.

julia> losses
2-element Array{Float64,1}:
 0.002439396192741583
 3.716891878724482e-7
```

In many applications to machine learning, "loss" will be an
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


## Verbose logging and inspecting control reports

The `IterationControl.train!` method can be given the keyword argument
`verbosity=...`, defaulting to `1`. The larger `verbosity`, the noisier.

The return value of `IterationControl.train!` is a tuple of `(control,
report)` tuples, where `report` is generated by `control` at the end
of training. For example, the final loss can be accessed from the
report of the `WithLossDo()` control:

```julia
model = SquareRooter(9)
reports = IterationControl.train!(model, Step(1), WithLossDo(println), NumberLimit(3));

julia> last(reports[2])
(loss = 0.1417301038062284, done = false, log = "")

julia> last(reports[2]).loss
  0.1417301038062284
```


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
`WithNumberDo(f=n->@info(n))`        | Call `f(n + 1)` where `n` is the number of complete control cycles so far  |`train!`                           | yes   |
`WithLossDo(f=x->@info("loss: $x"))` | Call `f(loss)` where `loss` is the current loss                            |`train!`, `loss`                   | yes   |
`WithTrainingLossesDo(f=v->@info(v))`| Call `f(v)` where `v` is the current batch of training losses              |`train!`, `training_losses`          | yes   |
`InvalidValue()`        | Stop when `NaN`, `Inf` or `-Inf` loss/training loss encountered                         |`train!`                           | yes   |
`Threshold(value=0.0)`  | Stop when `loss < value`                                                                |`train!`, `loss`                   | yes   |
`GL(alpha=2.0)`         | Stop after "Generalization Loss" exceeds `alpha`                                        |`train!`, `loss`                   | yes   | ``GL_α``
`Patience(n=5)`         | Stop after `n` consecutive loss increases                                               |`train!`, `loss`                   | yes   | ``UP_s``
`PQ(alpha=0.75, k=5)`   | Stop after "Progress-modified GL" exceeds `alpha`                                       |`train!`, `loss`, `training_losses`| yes   | ``PQ_α``
`Warmup(c; n=1)`        | Wait for `n` loss updates before checking criteria `c`                                  |`train!`                           | no    |
`Data(data)`            | Call `ingest!(model, item)` on the next `item` in the iterable `data`.                  |`train!`, `ingest!`                | yes   |

> Table 1. Atomic controls

**Stopping option.** All the following controls trigger a stop if the
provided function `f` returns `true` and `stop_if_true=true` is
specified in the constructor: `Callback`, `WithNumberDo`, `WithLossDo`,
`WithTrainingLossesDo`.

There are also three control wrappers to modify a control's behavior:

wrapper                                            | description
---------------------------------------------------|-------------------------------------------------------------------------
`IterationControl.skip(control; predicate=1)`      | Apply `control` every `predicate` applications of the control wrapper (can also be a function; see doc-string)
`IterationControl.louder(control; by=1)`           | Increase the verbosity level of `control` by the specified value (negative values lower verbosity)
`IterationControl.with_state_do(control; f=...)`   | Apply control *and* call `f(x)` where `x` is the internal state of control; useful for debugging. Default `f` logs state to `Info`. **Warning**: internal control state is not yet part of public API.
`IterationControl.composite(controls...)`          | Apply each `control` in `controls` in sequence; mostly for under-the-hood use

> Table 2. Wrapped controls


## Access to model through a wrapper

Note that functions ordinarily applied to `model` by some control
(e.g., a `Callback`) will instead be applied to
`IterationControl.expose(model)` if `IterationControl.expose` is
appropriately overloaded.


## Implementing new controls

There is no abstract control type; any object can be a
control. Behavior is implemented using a functional style interface
with six methods. Only the first two are compulsory (the fallbacks for
`done`, `takedown`, `needs_loss` and `needs_training_losses` always
return `false` and `NamedTuple()` respectively.):

```julia
update!(control, model, verbosity, n) -> state  # initialization
update!(control, model, verbosity, n, state) -> state
done(control, state)::Bool
takedown(control, verbosity, state) -> human_readable_named_tuple
```

Here `n` is the control cycle count, i.e., one more than the the
number of completed control cycles.

If it is nonsensical to apply `control` to any model for which
`loss(model)` has not been overloaded, and we want an error thrown
when this is attempted, then declare `needs_loss(control::MyControl) =
true` to take value true. Otherwise `control` is applied anyway, and
`loss`, if called, returns `nothing`.

A second trait `needs_training_losses(control)` serves an analogous
purpose for training losses.

Here's a simplified version of how `IterationControl.train!` calls
these methods:

```julia
function train!(model, controls...; verbosity::Int=1)

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

	# finalization:
	return takedown(control, verbosity, state)
end
```
