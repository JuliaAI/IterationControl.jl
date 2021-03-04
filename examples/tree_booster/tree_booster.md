```@meta
EditURL = "<unknown>/../examples/tree_booster/tree_booster.jl"
```

# Using IterativeControl to train a tree-booster on the iris data set

In this demonstration we show how to the controls in
[IterationControl.jl](https://github.com/ablaom/IterationControl.jl)
with an iterative
[MLJ](https://github.com/alan-turing-institute/MLJ.jl) model, using
our bare hands. (MLJ will ultimately provide its own canned
`IteratedModel` wrapper to make this more convenient and
compositional.)

```@example tree_booster
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

import MLJ
using IterationControl

using Statistics
using Random
Random.seed!(123)

MLJ.color_off()
```

Loading some data and splitting observation indices into test/train:

```@example tree_booster
X, y = MLJ.@load_iris;
train, test = MLJ.partition(eachindex(y), 0.7, shuffle=true)
```

Import an model type:

```@example tree_booster
Booster = MLJ.@load EvoTreeClassifier verbosity=0
```

Note that in MLJ a "model" is just a container for
hyper-parameters. The objects we will iterate here are MLJ
[*machines*](https://alan-turing-institute.github.io/MLJ.jl/dev/machines/);
these bind the model to train/test data and, in the case of
iterative models, can be trained using a warm-restart.

Creating a machine:

```@example tree_booster
mach = MLJ.machine(Booster(nrounds=1), X, y);
nothing #hide
```

Lifting MLJ's `fit!(::Machine)` method to `IterativeControl.train!`:

```@example tree_booster
function IterationControl.train!(mach::MLJ.Machine{<:Booster}, n::Int)
    mlj_model = mach.model
    mlj_model.nrounds = mlj_model.nrounds + n
    MLJ.fit!(mach, rows=train, verbosity=0)
end
```

Lifting the out-of-sample loss:

```@example tree_booster
function IterationControl.loss(mach::MLJ.Machine{<:Booster})
    mlj_model = mach.model
    yhat = MLJ.predict(mach, rows=test)
    return MLJ.log_loss(yhat, y[test]) |> mean
end
```

Iterating with controls:

```@example tree_booster
logging(mach) = "loss: $(IterationControl.loss(mach))"

IterationControl.train!(mach,
                        Train(5),
                        GL(),
                        Info(logging))
```

Continuing iteration with a different stopping criterion:

```@example tree_booster
IterationControl.train!(mach,
                        Train(5),
                        NumberLimit(10),
                        Info(logging))
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

