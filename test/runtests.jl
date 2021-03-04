using IterationControl
using Test

const IC = IterationControl

include("_models_for_testing.jl")

@testset "utilities" begin
    include("utilities.jl")
end

# this test must happen before test of controls.jl:
@testset "api" begin
     include("api.jl")
end

@testset "controls" begin
    include("controls.jl")
end

@testset "stopping_controls" begin
    include("stopping_controls.jl")
end

@testset "composite_controls" begin
    include("composite_controls.jl")
end

@testset "wrapped_controls" begin
    include("wrapped_controls.jl")
end

@testset "train!" begin
    include("train.jl")
end

