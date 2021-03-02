```@meta
EditURL = "<unknown>/iris.jl"
```

# Using IterativeControl to train a tree-booster on the iris data set

In this demonstration we show how to the controls in
[IterationControl.jl](https://github.com/ablaom/IterationControl.jl)
with an iterative
[MLJ](https://github.com/alan-turing-institute/MLJ.jl) model, using
our bare hands. (MLJ will ultimately provide its own canned
`IteratedModel` wrapper to make this more convenient and
compositional.)

```julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

import MLJ
using IterationControl

using Statistics
using Random
Random.seed!(123)

using Plots
pyplot(size=(600, 300*(sqrt(5)-1)));

MLJ.color_off()
```

```
false
```

Loading some data and splitting observation indices into test/train:

```julia
X, y = MLJ.@load_iris;
train, test = MLJ.partition(eachindex(y), 0.7, shuffle=true)
```

```
([125, 100, 130, 9, 70, 148, 39, 64, 6, 107, 73, 50, 4, 126, 116, 131, 121, 48, 94, 143, 149, 68, 89, 128, 25, 10, 56, 16, 7, 49, 82, 120, 42, 33, 19, 62, 103, 43, 1, 35, 88, 76, 104, 123, 87, 67, 66, 22, 28, 17, 119, 77, 141, 60, 136, 95, 23, 105, 69, 51, 53, 115, 3, 32, 142, 63, 15, 150, 75, 111, 132, 127, 86, 81, 29, 2, 113, 99, 38, 20, 138, 54, 11, 31, 117, 58, 55, 145, 65, 133, 84, 93, 146, 45, 8, 134, 114, 52, 74, 44, 61, 83, 18, 122, 26], [97, 78, 30, 108, 101, 24, 85, 91, 135, 96, 124, 92, 71, 102, 129, 27, 36, 46, 118, 57, 12, 90, 137, 98, 14, 13, 80, 37, 40, 79, 34, 110, 59, 139, 21, 112, 144, 140, 72, 109, 41, 106, 147, 47, 5])
```

Import an model type:

```julia
Booster = MLJ.@load EvoTreeClassifier verbosity=0
```

```
EvoTrees.EvoTreeClassifier
```

Note that in MLJ a "model" is just a container for
hyper-parameters. The objects we will iterate here are MLJ
[*machines*](https://alan-turing-institute.github.io/MLJ.jl/dev/machines/);
these bind the model to train/test data and, in the case of
iterative models, can be trained using a warm-restart.

Creating a machine:

```julia
mach = MLJ.machine(Booster(nrounds=1), X, y);
nothing #hide
```

Lifting MLJ's `fit!(::Machine)` method to `IterativeControl.train!`:

```julia
function IterationControl.train!(mach::MLJ.Machine{<:Booster}, n::Int)
    mlj_model = mach.model
    mlj_model.nrounds = mlj_model.nrounds + n
    MLJ.fit!(mach, rows=train, verbosity=0)
end
```

Lifting the out-of-sample loss:

```julia
function IterationControl.loss(mach::MLJ.Machine{<:Booster})
    mlj_model = mach.model
    yhat = MLJ.predict(mach, rows=test)
    return MLJ.log_loss(yhat, y[test]) |> mean
end
```

Iterating with controls:

```julia
logging(mach) = "loss: $(IterationControl.loss(mach))"

IterationControl.train!(mach,
                        Train(5),
                        GL(),
                        Info(logging))
```

```
((Train(5), NamedTuple()), (GL(2.0), (done = true, log = "Early stop triggered by GL(2.0) stopping criterion. ")), (Info{typeof(Main.##259.logging)}(Main.##259.logging), NamedTuple()))
```

Continuing iteration with a different stopping criterion:

```julia
IterationControl.train!(mach,
                        Train(5),
                        NumberLimit(10),
                        Info(logging))
```

```
((Train(5), NamedTuple()), (NumberLimit(10), (done = true, log = "Early stop triggered by NumberLimit(10) stopping criterion. ")), (Info{typeof(Main.##259.logging)}(Main.##259.logging), NamedTuple()))
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

