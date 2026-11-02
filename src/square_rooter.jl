# ## SQUARE ROOTER

# Consider a model to compute Babylonian approximations to a square root:

module SquareRooterExample

mutable struct SquareRooter
    x::Float64     # input - number to be square rooted
    root::Float64  # current approximation of root
    training_losses::Vector{Float64} # successive approximation differences
    SquareRooter(x) = new(x, 1.0, Float64[])
end

function train!(m::SquareRooter, Δn::Int)
    m.training_losses = Float64[]
    for i in 1:Δn
        next_guess = (m.root + m.x/m.root)/2
        push!(m.training_losses, abs(next_guess - m.root))
        m.root = next_guess
    end
end

loss(m::SquareRooter) = abs(m.root^2 - m.x)
training_losses(m::SquareRooter) = m.training_losses

export SquareRooter, train!, loss, training_losses

end

import .SquareRooterExample: SquareRooter
train!(m::SquareRooter, Δn::Int) = SquareRooterExample.train!(m, Δn)
loss(m::SquareRooter) = SquareRooterExample.loss(m)
training_losses(m::SquareRooter) = SquareRooterExample.training_losses(m)
