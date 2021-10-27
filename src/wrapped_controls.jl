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


# # WithStateDo

# helper:
shout(control, state) = @info "$(typeof(control)) state: $(flat(state))"

struct WithStateDo{C,F}
    control::C
    f::F
end

"""
    IterationControl.with_state_do(control,
                                  f=x->@info "\$(typeof(control)) state: \$x")

Wrap `control` to give access to it's internal state. Acts exactly
like `control` except that `f` is called on the internal state of
`control`. If `f` is not specified, the control type and state are
logged to `Info` at every update (useful for debugging new controls).

**Warning.** The internal state of a control is not yet considered
part of the public interface and could change between in any pre 1.0
release of IterationControl.jl.

"""
with_state_do(c; f=state->shout(c, state)) = WithStateDo(c, f)

# api:
for op in [:done, :takedown]
    eval(:($op(d::WithStateDo, args...) = $op(d.control, args...)))
end
@inline function update!(d::WithStateDo, args...)
    state = update!(d.control, args...)
    d.f(state)
    return state
end


# ## Skip

struct Skip{C,F<:Function}
    control::C
    predicate::F
end

_pred(predicate) = predicate
_pred(predicate::Int) = t -> mod(t, predicate) == 0

"""
    IterationControl.skip(control, predicate=1)

An iteration control wrapper.

If `predicate` is an **integer**, `k`: Apply `control` on every `k`
calls to apply the wrapped control, starting with the `k`th call.

If `predicate` is a **function**: Apply `control` as usual when
`predicate(n + 1)` is `true` but otherwise skip. Here `n` is the
number of control cycles applied so far.

"""
skip(control; predicate::Int=1) = Skip(control, _pred(predicate))

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

done(s::Skip, state) = done(s.control, state.atomic_state)

# can't be done if atomic state never intialized:
done(s::Skip, state::NamedTuple{(:n,)}) = false

takedown(s::Skip, verbosity, state) =
    takedown(s.control, verbosity, state.atomic_state)
takedown(::Skip, ::Any, ::NamedTuple{(:n,)}) = NamedTuple()
