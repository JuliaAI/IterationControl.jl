using IterationControl
import IterationControl:_repr, SquareRooter
import IterationControl as IC

@testset "helper for compact represtenation of columns" begin
    @test _repr("junk") == IterationControl.OMITTED
    @test _repr([[]]) == IterationControl.OMITTED
    @test _repr([1,2,3, 4]) == "[ …, 2, 3, 4]"
    @test _repr([1,2,3]) == "[1, 2, 3]"
    @test _repr([1,2]) == "[1, 2]"
    @test _repr(typeof(1)) == "<:Int64"
    @test _repr(typeof([1,])) == "<:Array"
end


# # GENERATE A TRACE

model = SquareRooter(4)
control = Threshold(0.1)
verbosity = 0

# train for one iteration:
IterationControl.train!(model, 1)

# initialize state and trace:
state = IterationControl.update!(control, model, verbosity, 1)
trace = IterationControl.ControlTrace(control, state)

# train another iteration:
IterationControl.train!(model, 1)

# update state and trace:
state = IterationControl.update!(control, model, verbosity, 1, state)
push!(trace, state)

@testset "property access" begin
    @test propertynames(trace) == (:loss, :excess, IterationControl.TRACE_FIELDS...)
    @test trace.control == control
    @test trace.loss == [2.25, 0.20249999999999968]
    @test trace.excess == [2.15, 0.10249999999999967]
    @test !isdefined(trace, :message)
    @test !trace.triggered_stop
    @test trace.table == (; loss=trace.loss, excess=trace.excess)
end

@testset "display" begin
    @test sprint(show, MIME("text/plain"), trace) ==
        """
        ControlTrace
          control: Threshold(0.1)
          loss: [2.25, 0.20249999999999968]
          excess: [2.15, 0.10249999999999967]
          table: (loss = …, excess = …)
          triggered_stop: false
        """ |> rstrip
    @test sprint(show, trace) ==
        "ControlTrace{<:Threshold}, ending (loss = 0.20249999999999968, "*
        "excess = 0.10249999999999967)"
end

# one more train and update will trigger a stop:
IterationControl.train!(model, 1)
state = IterationControl.update!(control, model, verbosity, 1, state)
push!(trace, state)

IC.takedown!(trace, verbosity, state)

@testset "display" begin
    @test sprint(show, MIME("text/plain"), trace) ==
        """
        ControlTrace
          control: Threshold(0.1)
          loss: [2.25, 0.20249999999999968, 0.002439396192741583]
          excess: [2.15, 0.10249999999999967, -0.09756060380725842]
          table: (loss = …, excess = …)
          message: Stop triggered by Threshold(0.1) stopping criterion. 
          triggered_stop: true
        """ |> rstrip
    @test sprint(show, trace) ==
        "ControlTrace{<:Threshold}, ending (loss = 0.002439396192741583, "*
        "excess = -0.09756060380725842)"
end


@testset "delayed addition of trace table (skip control)" begin
    model = SquareRooter(4)
    control = IC.skip(NumberLimit(3), predicate=2)
    verbosity = 0

    # control cycle #1

    # train one iteration:
    IterationControl.train!(model, 1)

    # initialize state and trace:
    state = IterationControl.update!(control, model, verbosity, 1)
    trace = IterationControl.ControlTrace(control, state)
    @test isempty(trace) # because of first skip

    # control cycle #2

    # training iteration:
    IterationControl.train!(model, 1)

    # update state and trace:
    state = IterationControl.update!(control, model, verbosity, 2, state)
    push!(trace, state)
    @test trace.table.number == [1,]

    # control cycle #3

    # train another iteration:
    IterationControl.train!(model, 1)

    # update state and trace:
    state = IterationControl.update!(control, model, verbosity, 3, state)
    push!(trace, state)
    @test trace.table.number == [1, 1]

    # control cycle #4

    # train another iteration:
    IterationControl.train!(model, 1)

    # update state and trace:
    state = IterationControl.update!(control, model, verbosity, 4, state)
    push!(trace, state)
    @test trace.table.number == [1, 1, 2]

    IC.takedown!(trace, verbosity, state)
    @test !trace.triggered_stop
end

true
