using Tables

import IterationControl as IC
import IterationControl.SquareRooter

@testset "Tables.jl interface for ControlTrace" begin
    model = SquareRooter(16)
    traces = IC.train!(model, Step(), NumberLimit(2), verbosity=-1)
    trace = traces[1]
    @test Tables.rowtable(trace) == [
        (new_iterations = 1, new_steps = 1),
        (new_iterations = 2, new_steps = 2),
    ]
    @test Tables.istable(trace)
    @test Tables.columnaccess(trace)
    @test Tables.columns(trace) == Tables.columntable(trace)
    @test Tables.columnindex(trace, :new_iterations) == 1
    @test Tables.columnindex(trace, "new_iterations") == 1
    @test Tables.getcolumn(trace, 1) == [1, 2]
    @test Tables.subset(trace, 1:2) == Tables.subset(trace.table, 1:2)
end

true
