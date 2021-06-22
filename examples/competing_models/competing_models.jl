# # A demonstration of controlling models competing in parallel

# Controlling competing iterative models in parallel is useful in
# optimization. For example, one might try several different
# optimization strategies in parallel, and divert resources to the
# strategy or strategies that are performing best.

# In this toy illustration, several anteaters of varying health
# compete for food from a common supply - a stream of random "ant" or
# inedible "seed" items. An anteater's `health` attribute is the
# probability he is able to successfully catch an ant when the
# source makes it available.

# If an anteater's score (the number of ants consumed) falls too far
# behind that of the leader, he stops consuming (his "training" stops
# early). An anteater also stops if he reaches the specified goal.

# Anteaters forage for food in parallel using Julia multi-threading.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# **Julia version** is assumed to be 1.6.*

using IterationControl
using .Threads
using Random
Random.seed!(123)
using Plots
pyplot(size = (600, 300*(sqrt(5) - 1)))

# ## Create a channel to supply food

N = 1_000_000
food = Channel{String}() do ch
    foreach(i -> put!(ch, rand(["ant", "seed"])), 1:N)
end

[take!(food) for i in 1:3]

# ## Define anteaters and how to "train" them

mutable struct Anteater
    health::Float64        # 0.0 = dead; 1.0 = excellent health
    n::Int                 # number of ants consumed
    Anteater(health) = new(health, 0)
end

function eat!(eater::Anteater)
    take!(food) == "ant" && rand() <= eater.health && (eater.n += 1)
    return eater
end

# "Training" for `n` iterations means take `n` items from the `food`
#  source and try to catch and eat each item if it is an "ant":

function IterationControl.train!(eater::Anteater, n)
    foreach(i->eat!(eater), 1:n)
    return eater
end


# ## Create a collection of competitors

eaters = [Anteater(0.7), Anteater(0.8), Anteater(0.9), Anteater(1.0)]

#-

n_eaters = length(eaters);

# For recording progress:
history = Float64[];

# ## Define the controls that will define the rules of the competition

const LEAD_TRIGGERING_DEFEAT = 5
const ANTS_TO_WIN = 100

scores(eaters) = [e.n for e in eaters]

is_too_far_behind(eater) =
    maximum(scores(eaters)) - eater.n > LEAD_TRIGGERING_DEFEAT

has_won(eater) = eater.n >= ANTS_TO_WIN

update_history(::Any) =  append!(history, scores(eaters))

controls = [Step(1),
            Callback(is_too_far_behind, stop_if_true=true),
            Callback(has_won, stop_if_true=true),
            Callback(update_history),
            NumberLimit(1000)]


# ## Run the competition

@sync for e in eaters
    Threads.@spawn IterationControl.train!(e, controls...)
end


# ## Plot the results

n_events = div(length(history), n_eaters)
history = reshape(history, n_eaters, n_events)'

plot(history,
     title="Ant eater competition",
     ylab="num of ants consumed",
     xlab="num food items taken from source")

using Literate #src
Literate.markdown(@__FILE__, @__DIR__, execute=false) #src
Literate.notebook(@__FILE__, @__DIR__, execute=true) #src
