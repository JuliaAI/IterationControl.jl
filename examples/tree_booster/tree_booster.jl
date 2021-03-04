# # Using IterativeControl to train a tree-booster on the iris data set

# In this demonstration we show how to the controls in
# [IterationControl.jl](https://github.com/ablaom/IterationControl.jl)
# with an iterative
# [MLJ](https://github.com/alan-turing-institute/MLJ.jl) model, using
# our bare hands. (MLJ will ultimately provide its own canned
# `IteratedModel` wrapper to make this more convenient and
# compositional.)

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

import MLJ
using IterationControl

using Statistics
using Random
Random.seed!(123)

MLJ.color_off()

# Loading some data and splitting observation indices into test/train:

X, y = MLJ.@load_iris;
train, test = MLJ.partition(eachindex(y), 0.7, shuffle=true)

# Import an model type:

Booster = MLJ.@load EvoTreeClassifier verbosity=0

# Note that in MLJ a "model" is just a container for
# hyper-parameters. The objects we will iterate here are MLJ
# [*machines*](https://alan-turing-institute.github.io/MLJ.jl/dev/machines/);
# these bind the model to train/test data and, in the case of
# iterative models, can be trained using a warm-restart.

# Creating a machine:

mach = MLJ.machine(Booster(nrounds=1), X, y);

# Lifting MLJ's `fit!(::Machine)` method to `IterativeControl.train!`:

function IterationControl.train!(mach::MLJ.Machine{<:Booster}, n::Int)
    mlj_model = mach.model
    mlj_model.nrounds = mlj_model.nrounds + n
    MLJ.fit!(mach, rows=train, verbosity=0)
end

# Lifting the out-of-sample loss:

function IterationControl.loss(mach::MLJ.Machine{<:Booster})
    mlj_model = mach.model
    yhat = MLJ.predict(mach, rows=test)
    return MLJ.log_loss(yhat, y[test]) |> mean
end

# Iterating with controls:

logging(mach) = "loss: $(IterationControl.loss(mach))"

IterationControl.train!(mach,
                        Train(5),
                        GL(),
                        Info(logging))

# Continuing iteration with a different stopping criterion:

IterationControl.train!(mach,
                        Train(5),
                        NumberLimit(10),
                        Info(logging))

using Literate #src
Literate.markdown(@__FILE__, @__DIR__, execute=false) #src
Literate.notebook(@__FILE__, @__DIR__, execute=true) #src
