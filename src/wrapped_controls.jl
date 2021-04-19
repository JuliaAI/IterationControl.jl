# # Louder

struct Louder{C}
    control::C
    by::Int64
end

"""
    IterationControl.louder(control, by=1)

Wrap `control` to make in more (or less) verbose. The same as
`control`, but as if the global `verbosity` were increased by the value
`by`.

"""
louder(c; by=1) = Louder(c, by)

# api:
done(d::Louder, state) = done(d.control, state)
update!(d::Louder, model, verbosity, args...) =
    update!(d.control, model, verbosity + d.by, args...)
takedown(d::Louder, verbosity, state) =
    takedown(d.control, verbosity + d.by, state)


# # Debug

struct Debug{C}
    control::C
end

"""
    IterationControl.debug(control)

Wrap `control` for debugging purposes. Acts exactly like `control`
except that the internal state of `control` is logged to `Info` at
every update.

"""
debug(c) = Debug(c)

# helper:
shout(control, state) = @info "`$(typeof(control))` state: $(flat(state))"

# api:
for f in [:done, :takedown]
    eval(:($f(d::Debug, args...) = $f(d.control, args...)))
end
@inline function update!(d::Debug, args...)
    state = update!(d.control, args...)
    shout(d.control, state)
    return state
end


# ## Skip

struct Skip{C,F<:Function}
    control::C
    predicate::F
end

_pred(predicate) = predicate
_pred(predicate::Int) = t -> mod(t + 1, predicate) == 0

"""
    IterationControl.skip(control, predicate=1)

An iteration control wrapper.

If `predicate` is an **integer**, `k`: Apply `control` on every `k`
calls to apply the wrapper, starting with the `k`th call.

If `predicate` is a **function**: Apply `control` as usual when
`predicate(n + 1)` is `true` but otherwise skip. Here `n` is the
number of calls to apply the wrapped control so far.

"""
skip(control; predicate::Int=1) = Skip(control, _pred(predicate))

_state(s, model, verbosity, n, atomic_state...) = if s.predicate(n)
    new_atomic_state = update!(s.control, model, verbosity + 1, atomic_state...)
    return (atomic_state = new_atomic_state, n = n + 1)
else
    return nothing
end

function update!(s::Skip, model, verbosity)
    state_candidate = _state(s, model, verbosity, 0)
    state_candidate isa Nothing && return (n = 1, )
    return state_candidate
end

# in case atomic state is not initialized in first `update` call:
function update!(s::Skip, model, verbosity, state::NamedTuple{(:n,)})
    state_candidate = _state(s, model, verbosity, state.n)
    state_candidate isa Nothing && return (n = state.n + 1, )
    return state_candidate
end

# regular update:
function update!(s::Skip, model, verbosity, state)
    state_candidate = _state(s, model, verbosity, state.n, state.atomic_state)
    state_candidate isa Nothing &&
        return (atomic_state = state.atomic_state, n = state.n + 1)
    return state_candidate
end

done(s::Skip, state) = done(s.control, state.atomic_state)

# can't be done if atomic state never intialized:
done(s::Skip, state::NamedTuple{(:n,)}) = false

takedown(s::Skip, verbosity, state) =
    takedown(s.control, verbosity, state.atomic_state)
takedown(::Skip, ::Any, ::NamedTuple{(:n,)}) = NamedTuple()
