using Tables

import IterationControl as IC
import IterationControl.SquareRooter

@testset "Tables.jl interface for ControlTrace" begin
    model = SquareRooter(16)
    traces = IC.train!(model, Step(), NumberLimit(2), verbosity=-1)
    @test Tables.rowtable(traces[1]) == [
        (new_iterations = 1, new_steps = 1),
        (new_iterations = 2, new_steps = 2),
    ]
end

true
