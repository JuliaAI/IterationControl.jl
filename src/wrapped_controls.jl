# # Louder

struct Louder{C}
    control::C
    by::Int64
end

"""
    IterationControl.louder(control, by=1)

Wrap `control` to make in more (or less) verbose. The same as `control`, but as if the
global `verbosity` were increased by the value `by`.

"""
louder(c; by=1) = Louder(c, by)

# api:
done(d::Louder, state) = done(d.control, state)
update!(d::Louder, model, verbosity, args...) =
    update!(d.control, model, verbosity + d.by, args...)
takedown(d::Louder, verbosity, state) =
    takedown(d.control, verbosity + d.by, state)
public_state(d::Louder, state) = public_state(d.control, state)

# # WithStateDo

# helper:
shout(control, public_state) =
    (@info "`$(typeof(control))` state: $(public_state)"; public_state)

struct WithStateDo{C,F}
    control::C
    f::F
end

"""
    IterationControl.with_state_do(control,f=x->(@info "`\$(typeof(control))` state: \$x; x)")

Wrap `control` to give access to the public form of its internal state (which is also the
object accumulated by the corresponding [`ControlTrace`](@ref) returned by
[`IterationControl.train!`](@ref)). Acts exactly like `control` except that `f` is called
on current state. If `f` is not specified, the control type and state are logged to `Info`
at every update. Useful for debugging custom controls.

"""
with_state_do(c; f=state->shout(c, state)) = WithStateDo(c, f)

# api:
for op in [:done, :takedown]
    eval(:($op(d::WithStateDo, args...) = $op(d.control, args...)))
end
@inline function update!(d::WithStateDo, args...)
    state = update!(d.control, args...)
    d.f(public_state(d.control, state))
    return state
end

public_state(d::WithStateDo, state) = public_state(d.control, state)


# ## Skip

struct Skip{C,F}
    control::C
    predicate::F
    step::Int
end

Base.show(io::IO, s::Skip) = s.step == 0 ?
    print(io, "skip($(s.control), …)") :
    print(io, "skip($(s.control), predicate=$(s.step))")
Base.show(io::IO, ::MIME"text/plain", s::Skip) = show(io, s)

accept(predicate) = t -> mod(t, predicate) == 0

"""
    IterationControl.skip(control, predicate=1)

An iteration control wrapper.

If `predicate` is an **integer**, `k`: Apply `control` on every `k`
calls to apply the wrapped control, starting with the `k`th call.

If `predicate` is a **callable**: Apply `control` as usual when
`predicate(n + 1)` is `true` but otherwise skip. Here `n` is the
number of control cycles applied so far.

While the application of the underlying control gets skipped, the (unchanged) internal
state is nevertheless recorded to the corresponding [`ControlTrace`](@ref), but this trace
does not start recording until the first application of `control`:

```julia-repl
using IterationControl
model = IterationControl.SquareRooter(4)

julia> traces = IC.train!(
    m,
    Step(1),
    IterationControl.skip(WithNumberDo(x->(@info(x); x)), predicate=2),
    NumberLimit(5),
)
[ Info: 2
[ Info: 4
[ Info: final loss: 0.0
[ Info: final training loss: 0.0
[ Info: Stop triggered by NumberLimit(5) stopping criterion.
3-element Vector{Any}:
 ControlTrace{<:Step}, ending (new_iterations = 5, new_steps = 5)
 ControlTrace{<:Skip}, ending (number = 4, called_back = 4)
 ControlTrace{<:NumberLimit}, ending (number = 5,)

julia> traces[2].number
4-element Vector{Int64}:
 2
 2
 4
 4
```
"""
function skip(control; predicate=1)
    predicate isa Integer && return Skip(control, accept(predicate), predicate)
    return Skip(control, predicate, 0)
end

_state(s, model, verbosity, n, atomic_state...) = if s.predicate(n)
    new_atomic_state =
        update!(s.control, model, verbosity + 1, n, atomic_state...)
    return (atomic_state = new_atomic_state, n = n)
else
    return nothing
end

function update!(s::Skip, model, verbosity, n)
    state_candidate = _state(s, model, verbosity, n)
    state_candidate isa Nothing && return (n = n, )
    return state_candidate
end

# in case atomic state is not initialized in first `update` call:
function update!(s::Skip, model, verbosity, n, state::NamedTuple{(:n,)})
    state_candidate = _state(s, model, verbosity, n)
    state_candidate isa Nothing && return (n = n, )
    return state_candidate
end

# regular update:
function update!(s::Skip, model, verbosity, n, state)
    state_candidate = _state(s, model, verbosity, n, state.atomic_state)
    state_candidate isa Nothing &&
        return (atomic_state = state.atomic_state, n = n)
    return state_candidate
end

public_state(::Skip, ::NamedTuple{(:n,)}) = nothing
public_state(s::Skip, state) = public_state(s.control, state.atomic_state)

done(s::Skip, state) = done(s.control, state.atomic_state)

# can't be done if atomic state never initialized:
done(s::Skip, state::NamedTuple{(:n,)}) = false

takedown(s::Skip, verbosity, state) =
    takedown(s.control, verbosity, state.atomic_state)
takedown(::Skip, ::Any, ::NamedTuple{(:n,)}) = ""
