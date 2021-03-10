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

If `predicate` is an **integer**, `n`: Apply `control` on every `n`
calls to apply the wrapper, starting with the `n`th call.

If `predicate` is a **function**: Apply `control` as usual when
`predicate(t + 1)` is `true` but otherwise skip. Here `t` is the
number of calls to apply the wrapper so far.

"""
skip(control; predicate::Int=1) = Skip(control, _pred(predicate))

_state(s, model, verbosity, t) = if s.predicate(t)
    atomic_state = update!(s.control, model, verbosity + 1)
    return (atomic_state = atomic_state, t = t + 1)
else
    return nothing
end

function update!(s::Skip, model, verbosity)
    state_candidate = _state(s, model, verbosity, 0)
    state_candidate isa Nothing && return (t = 1, )
    return state_candidate
end

# in case atomic state is not initialized in first `update` call:
function update!(s::Skip, model, verbosity, state::NamedTuple{(:t,)})
    state_candidate = _state(s, model, verbosity, state.t)
    state_candidate isa Nothing && return (t = state.t + 1, )
    return state_candidate
end

# regular update:
function update!(s::Skip, model, verbosity, state)
    state_candidate = _state(s, model, verbosity, state.t)
    state_candidate isa Nothing &&
        return (atomic_state = state.atomic_state, t = state.t + 1)
    return state_candidate
end

done(s::Skip, state) = done(s.control, state.atomic_state)

# can't be done if atomic state never intialized:
done(s::Skip, state::NamedTuple{(:t,)}) = false

takedown(s::Skip, verbosity, state) =
    takedown(s.control, verbosity, state.atomic_state)
takedown(::Skip, ::Any, ::NamedTuple{(:t,)}) = NamedTuple()
