using IterationControl
using Test

const IC = IterationControl

include("_models_for_testing.jl")
const SquareRooter = IC.SquareRooter

test_files = [
    "square_rooter.jl",
    "utilities.jl",
    "api.jl",
    "controls.jl",
    "stopping_controls.jl",
    "composite_controls.jl",
    "wrapped_controls.jl",
    "train.jl",
    "TablesExt.jl",
    "deprecated.jl",
]

files = isempty(ARGS) ? test_files : ARGS

for file in files
    quote
        @testset $file begin
            include($file)
        end
    end |> eval
end

