module IterationControl

import Base.*
import EarlyStopping: done, StoppingCriterion, StoppingCriterion
using InteractiveUtils
using EarlyStopping

const ES = EarlyStopping

# make a list of all controls:
const CONTROLS = [:Step,
                  :Info,
                  :Warn,
                  :Error,
                  :Callback,
                  :WithLossDo,
                  :WithTrainingLossesDo,
                  :WithNumberDo,
                  :Data]
for criterion in subtypes(StoppingCriterion)
    control = split(string(criterion), ".") |> last |> Symbol
    push!(CONTROLS, control)
end

# export controls:
for control in CONTROLS
    eval(:(export $control))
end

# re-export stopping criterion from EarlyStopping.jl:
for criterion in subtypes(StoppingCriterion)
    ex = split(string(criterion), ".") |> last |> Symbol
    eval(:(export $ex))
end

include("utilities.jl")
include("api.jl")
include("stopping_controls.jl")
include("composite_controls.jl")
include("wrapped_controls.jl")
include("controls.jl")
include("train.jl")

end # module
