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
