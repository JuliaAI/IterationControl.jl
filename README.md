# IterationControl.jl

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/ablaom/IterationControl.jl/workflows/CI/badge.svg)](https://github.com/ablaom/IterationControl.jl/actions)| [![codecov.io](http://codecov.io/github/ablaom/IterationControl.jl/coverage.svg?branch=master)](http://codecov.io/github/ablaom/IterationControl.jl?branch=master) |

Not registered and still experimental.
Well tested with complete doc-strings.

A package for controlling iterative algorithms.

To do: Add data interface point.


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
with any number of more sophisticated *controls* by "lifting" the
method `train!` to the `IterationControl.train!` method defined in
this package:

```julia
using IterativeControl
IterationControl.train!(model, n) =  train!(model, n)
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
julia> IterationControl.train!(model, Train(2), NumberLimit(3), Info(m->m.root));
[ Info: 3.4
[ Info: 3.00009155413138
[ Info: 3.0
[ Info: Early stop triggered by NumberLimit(3) stopping criterion.
```

If `model` admits a method returning a loss (for example, the
difference between `x` and the square of `root`), then we can lift
that method to `IterationControl.loss` to enable control using
loss-based stopping criteria, such as a threshold:

```julia
model = SquareRooter(4)
train!(model, 1)

julia> loss(model)
2.25

IterationControl.loss(model) = loss(model) # lifting

julia> IterationControl.train!(model, Train(1), Threshold(0.0001), Info(loss));
julia> IterationControl.train!(model, Train(1), Threshold(0.0001), Info(loss));
[ Info: 0.20249999999999968
[ Info: 0.002439396192741583
[ Info: 3.716891878724482e-7
[ Info: Early stop triggered by Threshold(0.0001) stopping criterion.
```

If training `model` generates user-inspectable "training losses" (one
per iteration), then lifting the appropriate access function to
`IterationControl.training_losses` enables Prechelt's
progress-modified generalization loss stopping criterion, `PQ`, the
only criterion from the
[EarlyStopping.jl](https://github.com/ablaom/EarlyStopping.jl) package
not otherwise enabled (with a regular loss method lifted to
`IterationControl.loss`).

*Reference.* [Prechelt, Lutz
 (1998)](https://link.springer.com/chapter/10.1007%2F3-540-49430-8_3):
 "Early Stopping - But When?", in *Neural Networks: Tricks of the
 Trade*, ed. G. Orr, Springer.

     
## Controls provided

Controls are repetitively applied until a stopping criterion is
triggered. The first control in a call
`IterativeControl.train!(models, controls...)` is ordinarily of type
`Train`. Each control type has a detailed doc-string. Here is short
summary (some advanced options not included):

control                 | description                                                               | notation in Prechelt
------------------------|---------------------------------------------------------------------------|---------------------
`Train(n=1)`            | Train model for `n` iterations                                            |
`Info(f=identity)`      | Log to `Info` the value of `f(model)`                                     |
`Warn(predicate, f="")` | Log to `Warn` the value of `f` or `f(model)` if `predicate(model)` holds  |
`Error(predicate, f="")`| Log to `Error` the value of `f` or `f(model)` if `predicate(model)` holds and stop |
`Callback(f=_->nothing)`| Call `f(model)`
`Never()`               | Never stop                                                                |
`NotANumber()`          | Stop when `NaN` encountered                                               |
`TimeLimit(t=0.5)`      | Stop after `t` hours                                                      |
`NumberLimit(n=100)`    | Stop after `n` loss updates (excl. "training losses")                     |
`Threshold(value=0.0)`  | Stop when `loss < value`                                                  |
`GL(alpha=2.0)`         | Stop after "Generalization Loss" exceeds `alpha`                          | ``GL_α``
`PQ(alpha=0.75, k=5)`   | Stop after "Progress-modified GL" exceeds `alpha`                         | ``PQ_α``
`Patience(n=5)`         | Stop after `n` consecutive loss increases                                 | ``UP_s``
   
> Table 1. Atomic controls

There are also three methods to modify a control's behavior:

wrapper                                            | description
---------------------------------------------------|-------------------------------------------------------------------------
`IterationControl.skip(control, predicate=1)`      | Apply `control` every `predicate` applications of the control wrapper (can also be a function; see doc-string)
`IterationControl.debug(control)`                  | Apply `control` but also log its state to `Info` (at any `verbosity` level)
`IterationControl.composite(controls...)`          | Apply each `control` in `controls` in sequence
