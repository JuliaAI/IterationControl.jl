module IterationControl

import Base.*
import EarlyStopping: done, StoppingCriterion, StoppingCriterion
using InteractiveUtils
using EarlyStopping

const ES = EarlyStopping

# controls:
export Train, Info, Warn, Error, Callback


# re-export stopping criterion from EarlyStopping.jl:
for criterion in subtypes(StoppingCriterion)
    ex = split(string(criterion), ".") |> last |> Symbol
    eval(:(export $ex))
end

include("utilities.jl")
include("api.jl")
include("controls.jl")
include("composite_controls.jl")
include("wrapped_controls.jl")
include("train.jl")

end # module
