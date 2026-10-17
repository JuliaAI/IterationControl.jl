# IterationControl.jl

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
Pkg.add("Tables") # optional
```

## Quick start

Suppose you have some kind of object `SquareRooter(x)`, for iteratively computing
approximations to the square root of `x`:

```@setup quick
using IterationControl.SquareRooterExample
```

```@example quick
model = SquareRooter(9)
model.root
```

```@example quick
train!(model, 2) # train for 2 iterations
model.root
```

```@example quick
train!(model, 1) # train for 1 more iteration
model.root
```

Then you can replace the integer argument `n` in `train!(model, n)`
with a number of more sophisticated **controls** by "lifting" the method
`train!` to the `IterationControl.train!` method defined in this
package:

```@example quick
using IterationControl
IterationControl.train!(model::SquareRooter, n::Int) =  train!(model, n) # lifting
```
By definition, the lifted `train!` has the same functionality as the original one:

```@example quick
model = SquareRooter(9)
IterationControl.train!(model, 2)
model.root
```

But now we can also do this:

```@example quick
model = SquareRooter(9)
IterationControl.train!(model, Step(2), NumberLimit(3), Info(m->m.root));
nothing # hide
```

Here each control is repeatedly applied in sequence until one of them triggers a stop:

- The first control `Step(2)` says, "Train the model two more iterations"
- The `NumberLimit(3)` control asks, "Have I been applied 3 times yet?", signaling a stop
  (at the end of the current control cycle) if so
- The `Info` control logs the value of the function `m -> m.root`, evaluated on
  `model`, to `Info`. 

In this example only the `NumberLimit(3)` control can terminate model iteration.

!!! info 
    
	To try out the example above, you can use a built-in version of `SquareRooter`. Before
    executing the code above, run `using IterationControl.SquareRooterExample`.

The [Controls](ref) section documents all the controls provided by IterationControl.jl,
including those from EarlyStopping.jl.


## Loss-based stopping criteria

If `model` admits a method returning a loss then we can lift that method to
`IterationControl.loss` to enable control using a loss-based stopping criteria, such as a
loss threshold. Let's demonstrate this for the built-in square rooting example, for which
`loss` returns the difference between the root approximation and the actual 64-bit
precision square root:

```@example quick
using IterationControl
using IterationControl.SquareRooterExample
model = SquareRooter(4)
train!(model, 1)
loss(model)
```

```@example quick
IterationControl.loss(model::SquareRooter) = loss(model) # lifting
traces = IterationControl.train!(
    model, 
	Step(1),
	Threshold(0.0001), 
	Callback(m->m.root),
	WithLossDo(),
);
nothing # hide
```

```@repl quick
traces
```

## Control traces

In the demonstration above we used `Callback` instead of `Info` to suppress the logging of
the root value to `stdout`, and we added a special callback, `WithLossDo()`, which calls
the `loss` function on `model`. (More generally, `WithLossDo(f)` applies `f` to each
loss.) The return values of callbacks can be inspected from the **control traces**
returned by [`IterationControl.train!`](@ref):

```@example quick
step, threshold, callback, withlossdo = traces;
nothing # hide
```

```@repl quick
callback
```

```@repl quick
withlossdo
```

```@repl quick
withlossdo.loss
```

Control traces have a property called `table`, returning a named tuple of vectors, but
they also implement the [Tables.jl](https://tables.juliadata.org/stable/) interface,
allowing conversion to common Julia table formats:

```@repl quick
threshold.table
```

```@repl quick
using DataFrames
DataFrame(threshold)
```

It is possible to inspect traces "live" by running [`IterationControl.train!`](@ref) in an
asynchronous task. See the document string for details.


## Training losses

In many applications to machine learning, "loss" will be an out-of-sample loss, computed
after some iterations. If `model` additionally generates user-inspectable "training
losses" (one per iteration) then similarly lifting the appropriate access function to
`IterationControl.training_losses` enables Prechelt's progress-modified generalization
loss stopping criterion, [`PQ`](@ref).

`PQ` is the only criterion from the
[EarlyStopping.jl](https://github.com/ablaom/EarlyStopping.jl) package not otherwise
enabled when `IterationControl.loss` is overloaded as above.

*Reference.* [Prechelt, Lutz
 (1998)](https://link.springer.com/chapter/10.1007%2F3-540-49430-8_3): "Early Stopping -
 But When?", in *Neural Networks: Tricks of the Trade*, ed. G. Orr, Springer.

The interface just described is sufficient for controlling conventional machine learning
models with an iteration parameter, as [this](/examples/tree_booster/) tree boosting
example shows.


## Online and incremental training

For online or incremental training, lift the method for ingesting data into the model to
[`IterationControl.ingest!(model, datum)`](@ref) and use the control `Data(data)`. Here
`data` is any iterator generating the `datum` items to be ingested (one per application of
the control). By default, the [`Data`](@ref) control becomes passive after `data` is
exhausted. See [Access to model through a wrapper](@ref) below on dealing with any model
wrapping necessary to implement data ingestion.

A simple particle tracking example is given [here](/examples/particle/).


## Access to model through a wrapper

Note that functions ordinarily applied to `model` by some control (e.g., a `Callback`)
will instead be applied to [`IterationControl.expose(model)`](@ref) if this function has
been appropriately overloaded.


## Reference

- [`IterationControl.train!`](@ref)
- [`IterationControl.loss`](@ref)
- [`IterationControl.training_losses`](@ref)
- [`IterationControl.expose`](@ref)
- [`IterationControl.ControlTrace`](@ref)


```@docs
IterationControl.train!
IterationControl.loss
IterationControl.training_losses
IterationControl.ingest!
IterationControl.expose
IterationControl.ControlTrace
```
