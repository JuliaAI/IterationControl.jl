# # Early Stopping for a Tree Booster

# > Implementing early stopping for an iterative [MLJ](https://juliaml.ai) model.

# In actual practice, an MLJ user will use
# [`IteratedModel`](https://juliaai.github.io/MLJ.jl/stable/controlling_iterative_models/)
# wrapper, which is built on top of IterationControl.jl, because this is more convenient
# and compositional.

# The source for this demonstration can be found
# [here](https://github.com/JuliaAI/IterationControl.jl/tree/dev/docs/src/notebooks/tree_booster/).


import MLJBase
import EvoTrees.EvoTreeClassifier
using IterationControl
import StatisticalMeasures.log_loss

using Statistics
using Random
Random.seed!(123)

# Loading some data and splitting observation indices into test/train:

X, y = MLJBase.@load_iris; # a table and a vector
train, test = MLJBase.partition(eachindex(y), 0.7, shuffle=true);

# Instantiate an MLJ model:

booster = EvoTreeClassifier(nrounds=1)
Booster = typeof(booster);

# Note that in MLJ a "model" is just a container for hyper-parameters. The objects we will
# iteratively train here are MLJ
# [*machines*](https://JuliaAI.github.io/MLJBase.jl/dev/machines/); these bind the model
# to train/test data and, in the case of iterative models, can be trained using a
# warm-restart.

# Creating a machine:

mach = MLJBase.machine(booster, X, y);

# Lifting MLJ's `fit!(::Machine)` method to `IterativeControl.train!`:

function IterationControl.train!(mach::MLJBase.Machine{<:Booster}, n::Int)
    mlj_model = mach.model
    mlj_model.nrounds = mlj_model.nrounds + n
    MLJBase.fit!(mach, rows=train, verbosity=0)
end;

# Lifting the out-of-sample loss:

function IterationControl.loss(mach::MLJBase.Machine{<:Booster})
    mlj_model = mach.model
    yhat = MLJBase.predict(mach, rows=test)
    return log_loss(yhat, y[test])
end;

# Iterating with controls:

IterationControl.train!(
    mach,
    Step(5),
    GL(),
    Info(IterationControl.loss),
);

# Continuing iteration with a different stopping criterion:

IterationControl.train!(
    mach,
    Step(5),
    NumberLimit(10),
    Info(IterationControl.loss),
);

