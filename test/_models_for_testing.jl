# # DUMMY MODELS FOR TESTING


# ## SQUARE ROOTER

# Consider a model to compute Babylonian approximations to a square root:

mutable struct SquareRooter
    x::Float64     # input - number to be square rooted
    root::Float64  # current approximation of root
    training_losses::Vector{Float64} # successive approximation differences
    SquareRooter(x) = new(x, 1.0, Float64[])
end

function IterationControl.train!(m::SquareRooter, Δn::Int)
    m.training_losses = Float64[]
    for i in 1:Δn
        next_guess = (m.root + m.x/m.root)/2
        push!(m.training_losses, abs(next_guess - m.root))
        m.root = next_guess
    end
end

IterationControl.loss(m::SquareRooter) = abs(m.root^2 - m.x)
IterationControl.training_losses(m::SquareRooter) = m.training_losses

model = SquareRooter(4.0)
IterationControl.train!(model, 1)
@assert model.root ≈ 2.5
@assert IterationControl.loss(model) ≈ 25/4 - 4
IterationControl.train!(model, 100)
@assert IterationControl.loss(model) ≈ 0
@assert IterationControl.training_losses(model)[1:2] ≈
    abs.([41/20 - 5/2, 3281/1640 - 41/20])


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

model = Particle()
ingest!(model, 1)
train!(model, 1)
@assert loss(model) ≈ 0.9
ingest!(model, -0.9)
train!(model, 1)
@assert loss(model) ≈ 0.9

model = Particle()
ingest!(model, 1)
train!(model, 2)
@assert training_losses(model) ≈ [0.9, 0.81]
