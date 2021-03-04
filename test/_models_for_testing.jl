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


# ## PARTICLE TRACKER

# Consider a model that tracks a particle in one dimension, moving at
# a speed that is corrected whenever a new target position is injected
# into the model. The velocity correction is the difference between
# current position and the newly reported one, multiplied by a
# fixed learning rate.

mutable struct WhatsNext
    position::Float64
    velocity::Float64
    η::Float64 # learning rate
    training_losses::Vector{Float64} # successive speeds
    WhatsNext(position) = new(position, 0.0, 0.1, Float64[])
end

WhatsNext(; position=0.0) = WhatsNext(0.0)

# native train!
train!(model::WhatsNext) = model.position = model.position + model.velocity
function train!(model, n)
    training_losses = map(1:n) do _
        train!(model)
        abs(model.velocity)
    end
    model.training_losses = training_losses
    return nothing
end
