# # DUMMY MODELS FOR TESTING

# ## PARTICLE TRACKER (without methods lifted)

# Consider an object that tracks a particle in one dimension, moving,
# in discrete time, at a speed proportional to the distance away from
# some moving target. The particle is initially at rest at the origin,
# which is where the target also begins.

# Calling `train!` on the object moves it along for the specified
# number of time steps, while calling `ingest!` updates the target
# position.

mutable struct Particle
    position::Float64
    target::Float64
    velocity::Float64
    η::Float64 # learning rate
    training_losses::Vector{Float64}
    Particle(η) = new(0.0, 0.0, 0.0, η, Float64[])
end

Particle(; η=0.1) = Particle(η)

# native train!
function train!(model::Particle)
    model.velocity = model.η*(model.target - model.position)
    model.position = model.position + model.velocity
end

loss(model::Particle) =  abs(model.target - model.position)

function train!(model, n)
    training_losses = map(1:n) do _
        train!(model)
        loss(model)
    end
    model.training_losses = training_losses
    return nothing
end

training_losses(model::Particle) = model.training_losses

function ingest!(model::Particle, target)
    model.target = target
    return nothing
end

# temporary testing area

# model = Particle()
# ingest!(model, 1)
# train!(model, 1)
# @assert loss(model) ≈ 0.9
# ingest!(model, -0.9)
# train!(model, 1)
# @assert loss(model) ≈ 0.9

# model = Particle()
# ingest!(model, 1)
# train!(model, 2)
# @assert training_losses(model) ≈ [0.9, 0.81]
