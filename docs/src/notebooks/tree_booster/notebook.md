```@meta
EditURL = "notebook.jl"
```

# Early Stopping for a Tree Booster

> Implementing early stopping for an iterative [MLJ](https://juliaml.ai) model.

In actual practice, an MLJ user will use
[`IteratedModel`](https://juliaai.github.io/MLJ.jl/stable/controlling_iterative_models/)
wrapper, which is built on top of IterationControl.jl, because this is more convenient
and compositional.

The source for this demonstration can be found
[here](https://github.com/JuliaAI/IterationControl.jl/tree/dev/docs/src/notebooks/tree_booster/).

````@julia
import MLJBase
import EvoTrees.EvoTreeClassifier
using IterationControl
import StatisticalMeasures.log_loss

using Statistics
using Random
Random.seed!(123)
````

````
Random.TaskLocalRNG()
````

Loading some data and splitting observation indices into test/train:

````@julia
X, y = MLJBase.@load_iris; # a table and a vector
train, test = MLJBase.partition(eachindex(y), 0.7, shuffle=true);
````

Instantiate an MLJ model:

````@julia
booster = EvoTreeClassifier(nrounds=1)
Booster = typeof(booster);
````

Note that in MLJ a "model" is just a container for hyper-parameters. The objects we will
iteratively train here are MLJ
[*machines*](https://JuliaAI.github.io/MLJBase.jl/dev/machines/); these bind the model
to train/test data and, in the case of iterative models, can be trained using a
warm-restart.

Creating a machine:

````@julia
mach = MLJBase.machine(booster, X, y);
````

Lifting MLJ's `fit!(::Machine)` method to `IterativeControl.train!`:

````@julia
function IterationControl.train!(mach::MLJBase.Machine{<:Booster}, n::Int)
    mlj_model = mach.model
    mlj_model.nrounds = mlj_model.nrounds + n
    MLJBase.fit!(mach, rows=train, verbosity=0)
end;
````

Lifting the out-of-sample loss:

````@julia
function IterationControl.loss(mach::MLJBase.Machine{<:Booster})
    mlj_model = mach.model
    yhat = MLJBase.predict(mach, rows=test)
    return log_loss(yhat, y[test])
end;
````

Iterating with controls:

````@julia
IterationControl.train!(
    mach,
    Step(5),
    GL(),
    Info(IterationControl.loss),
);
````

````
[ Info: 0.39655590607605007
[ Info: 0.25803830912951475
[ Info: 0.20369477186348553
[ Info: 0.17930965646052324
[ Info: 0.15988563045516432
[ Info: 0.14971082736701255
[ Info: 0.1419051277333237
[ Info: 0.13624241869102463
[ Info: 0.12996659052675014
[ Info: 0.12758077552017272
[ Info: 0.1241560953359005
[ Info: 0.12118229064975733
[ Info: 0.11805805490577778
[ Info: 0.12337760263694478
[ Info: final loss: 0.12337760263694478
[ Info: Stop triggered by EarlyStopping.GL(2.0) stopping criterion. 

````

Continuing iteration with a different stopping criterion:

````@julia
IterationControl.train!(
    mach,
    Step(5),
    NumberLimit(10),
    Info(IterationControl.loss),
);
````

````
[ Info: 0.1284682197583394
[ Info: 0.1333389660968577
[ Info: 0.13635253372234196
[ Info: 0.14081266985093113
[ Info: 0.14362910766700177
[ Info: 0.1463502830758916
[ Info: 0.14873864305645823
[ Info: 0.15129491860311206
[ Info: 0.154195294592051
[ Info: 0.15754958515966186
[ Info: final loss: 0.15754958515966186
[ Info: Stop triggered by EarlyStopping.NumberLimit(10) stopping criterion. 

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

