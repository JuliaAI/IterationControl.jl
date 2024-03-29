struct CompositeControl{A,B}
    a::A
    b::B
    function CompositeControl(a::A, b::B) where {A, B}
        # `_in` method extended post-facto
        a isa StoppingCriterion && _in(a, b) && return b
        b isa StoppingCriterion && _in(b, a) && return a
        return new{A,B}(a, b)
    end
end
CompositeControl() = Never()
CompositeControl(a) = a
CompositeControl(a, b, c...) = CompositeControl(CompositeControl(a,b), c...)


"""
    composite(controls...)

Construct an iteration control that applies the specified `controls`
in sequence.

"""
composite(controls...) = CompositeControl(controls...)

update!(c::CompositeControl, m, v, n) =
    (a = update!(c.a, m, v, n), b = update!(c.b, m, v, n))
update!(c::CompositeControl, m, v, n, state) =
    (a = update!(c.a, m, v, n, state.a),
     b = update!(c.b, m, v, n, state.b))


# # RECURSION TO FLATTEN A CONTROL OR ITS STATE

flat(state) = (state,)
flat(state::NamedTuple{(:a,:b)}) = tuple(flat(state.a)..., flat(state.b)...)
flat(d::CompositeControl) = tuple(flat(d.a)..., flat(d.b)...)

_in(::Never, ::Any) = true
_in(::Never, ::CompositeControl) = true
_in(c1::Any, c2::Any) = c1 == c2
_in(c::Any, d::CompositeControl) = c in flat(d)
_in(::CompositeControl, ::Any) = false


# # DISPLAY

function Base.show(io::IO, c::CompositeControl)
    list = join(string.(flat(c)), ", ")
    print(io, "CompositeControl($list)")
end


# # RECURSION TO DEFINE `done`

# fallback for atomic controls:
_done(control, state, old_done) = old_done || done(control, state)
%
# composite:
_done(c::CompositeControl, state, old_done) =
    _done(c.a, state.a, _done(c.b, state.b, old_done))

done(c::CompositeControl, state) = _done(c, state, false)


# # RECURSION TO DEFINE `takedown`

# fallback for atomic controls:
function _takedown(control, v, state, old_takedown)
    new_takedown = (control, takedown(control, v, state))
    return tuple(old_takedown..., new_takedown)
end

# composite:
_takedown(c::CompositeControl, v, state, old_takedown) =
    _takedown(c.b, v, state.b,
              _takedown(c.a, v, state.a, old_takedown))

takedown(c::CompositeControl, v, state) = _takedown(c, v, state, ())


# # TRAITS

for ex in [:needs_loss, :needs_training_losses]
    quote
        $ex(c::CompositeControl) = any($ex, flat(c))
    end |> eval
end
