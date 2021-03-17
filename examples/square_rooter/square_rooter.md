```@meta
EditURL = "<unknown>/square_rooter.jl"
```

Introductory demonstration of IterationControl.jl

```julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
```

```
 Activating environment at `~/Dropbox/Julia7/MLJ/IterationControl/examples/square_rooter/Project.toml`

```

Here's a simple iterative mdel that computes Babylonian
approximations to a square root:

```julia
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
```

```
training_losses (generic function with 1 method)
```

And here it in action:

```julia
model = SquareRooter(9)
model.root
```

```
1.0
```

```julia
train!(model, 2) # train for 2 iterations
model.root
```

```
3.4
```

```julia
train!(model, 1) # train for 1 more iteration
model.root
```

```
3.023529411764706
```

 Then we can replace the integer argument `n` in `train!(model, n)`
 with a number of more sophisticated *controls* by "lifting" the method
`train!` to the `IterationControl.train!` method defined in this
package:

```julia
using IterationControl
IterationControl.train!(model::SquareRooter, n) =  train!(model, n)
```

The lifted `train!` has the same functionality as the original one:

```julia
model = SquareRooter(9)
IterationControl.train!(model, 2)

model.root
```

```
3.4
```

But now we can also do this:

```julia
IterationControl.train!(model, Step(2), NumberLimit(3), Info(m->m.root));
nothing #hide
```

```
┌ Info: 3.00009155413138
└ @ IterationControl /Users/anthony/.julia/packages/IterationControl/h3s8z/src/controls.jl:47
┌ Info: 3.0
└ @ IterationControl /Users/anthony/.julia/packages/IterationControl/h3s8z/src/controls.jl:47
┌ Info: 3.0
└ @ IterationControl /Users/anthony/.julia/packages/IterationControl/h3s8z/src/controls.jl:47
┌ Info: Stop triggered by NumberLimit(3) stopping criterion. 
└ @ IterationControl /Users/anthony/.julia/packages/IterationControl/h3s8z/src/stopping_controls.jl:75

```

Here each control is repeatedly applied until one of them triggers a
stop. The first control `Step(2)` says "train the model two more
iterations"; the second says "stop after 3 repetitions" (of the
sequence of control applications); and the third, "log the value of
the root to `Info`".

If `model` admits a method returning a loss (for example, the
difference between `x` and the square of `root`), then we can lift
that method to `IterationControl.loss` to enable control using
loss-based stopping criteria, such as a loss threshold. In the
demonstation below, we also include a callback:

```julia
model = SquareRooter(4)
train!(model, 1)
loss(model)
```

```
2.25
```

```julia
IterationControl.loss(model::SquareRooter) = loss(model)

losses = Float64[]
callback(model) = push!(losses, loss(model))

IterationControl.train!(model,
                        Step(1),
                        Threshold(0.0001),
                        Callback(callback));
nothing #hide
```

```
┌ Info: Stop triggered by Threshold(0.0001) stopping criterion. 
└ @ IterationControl /Users/anthony/.julia/packages/IterationControl/h3s8z/src/stopping_controls.jl:75

```

```julia
losses
```

```
3-element Array{Float64,1}:
 0.20249999999999968
 0.002439396192741583
 3.716891878724482e-7
```

If training `model` generates user-inspectable "training losses" (one
per iteration) then similarly lifting the appropriate access function
to `IterationControl.training_losses` enables Prechelt's
progress-modified generalization loss stopping criterion, `PQ`.

`PQ` is the only criterion from the
[EarlyStopping.jl](https://github.com/ablaom/EarlyStopping.jl) package
not otherwise enabled when `IterationControl.loss` is overloaded as
above.

*Reference.* [Prechelt, Lutz
 (1998)](https://link.springer.com/chapter/10.1007%2F3-540-49430-8_3):
 "Early Stopping - But When?", in *Neural Networks: Tricks of the
 Trade*, ed. G. Orr, Springer.

The interface just described is sufficient for controlling
conventional machine learning models with an iteration parameter, as
this [tree boosting
example](https://github.com/ablaom/IterationControl.jl/tree/master/examples/tree_booster)
shows.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

