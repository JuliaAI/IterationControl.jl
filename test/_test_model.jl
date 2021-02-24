## DUMMY REGRESSOR FOR TESTING

# dummy regressor returns the value of the training target at the mode of
# the training inputs, scaled by a factor determined by the
# "iteration" parameter `n`, this scaling approaching one as `n` goes
# to `Inf`. Inputs are `Count` vectors, target is `Continuous` vector.

module TestModel

using MLJBase
export Regressor

almost_one(n) = (1 + 1/n)

mutable struct Regressor <: MLJBase.Deterministic
    n::Int
end

Regressor(; n=10) = Regressor(n)

function MLJBase.fit(::Regressor, verbosity::Int, x, y)
    θ = y[findmax(x) |> last] # value of y at mode of x
    cache     = nothing
    report    = NamedTuple()
    return θ, cache, report
end

function MLJBase.predict(regressor::Regressor, θ, xnew)
    θ_n = θ*almost_one(regressor.n)
    return fill(θ_n, MLJBase.nrows(xnew))
end

MLJBase.iteration_parameter(::Type{<:Regressor}) = :n
MLJBase.input_scitype(::Type{<:Regressor}) = AbstractVector{Count}
MLJBase.target_scitype(::Type{<:Regressor}) =
    AbstractVector{MLJBase.Continuous}

end

using .TestModel

x = [1, 2]
y = [0.0, 10.0]
mach = model(Regressor(n=10), x, y)
fit!(mach, verbosity=0)
@assert predict(mach, x) ≈ [11.0, 11.0]
