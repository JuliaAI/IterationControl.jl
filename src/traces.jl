"""
    columntable(rows)


**Private method.**

Convert a tuple of named-tuples sharing the same keys into a named-tuple of vectors.

"""
function columntable(rows)
    names = keys(first(rows))
    columns = collect.(zip(values.(rows)...))
    NamedTuple{names}(columns)
end

_last(columntable) =
    NamedTuple{keys(columntable)}(last.(values(columntable)))

# NOTE. because of `skip` controls, we have to allow for the possiblity that
# `public_state(control, state) = `nothing`, so that construction of the table is delayed.

mutable struct ControlTrace{C}
    control::C
    triggered_stop::Bool
    table  # named tuple of vectors or `nothing`
    message::String
    function ControlTrace(control::C, state) where C
        triggered_stop = done(control, state)
        public_state = IterationControl.public_state(control, state)
        # if public state available, create tabel with one row:
        table = isnothing(public_state) ? nothing :
            columntable((public_state,))
        return new{C}(control, triggered_stop, table)
    end
end

Base.isempty(trace::ControlTrace) = isnothing(trace.table)

const TRACE_FIELDS = (:control, :table, :message, :triggered_stop)

function Base.propertynames(trace::ControlTrace)
    tablekeys = isnothing(trace.table) ? () : keys(trace.table)
    return (tablekeys...,  TRACE_FIELDS...)
end
function Base.getproperty(trace::ControlTrace, name::Symbol)
    name in TRACE_FIELDS && return getfield(trace, name)
    name in propertynames(trace) || throw(FieldError(typeof(trace), name))
    return getproperty(trace.table, name)
end

function Base.push!(trace::ControlTrace, state; update_only::Bool=false)
    public_state = IterationControl.public_state(trace.control, state)
    isnothing(public_state) && return trace
    if isnothing(trace.table)
        trace.triggered_stop = done(trace.control, state)
        # create a column table with one row:
        trace.table = columntable((public_state,))
        return trace
    end
    for name in keys(public_state)
        column = getproperty(trace.table, name)
        if update_only
            column[end] = getproperty(public_state, name)
        else
            push!(column, getproperty(public_state, name))
        end
    end
    trace.triggered_stop = done(trace.control, state)
    return trace
end

update!(trace::ControlTrace, state) =  push!(trace, state; update_only=true)

function takedown!(trace::ControlTrace, verbosity, state)
    trace.message = takedown(trace.control, verbosity,state)
    return trace
end


# Display helper
const OMITTED = "⟨omittted⟩"
_repr(x) = OMITTED
const Displayable = Union{Number,AbstractString,Char,Symbol}
function _repr(v::Vector{<:Displayable})
    length(v) ≤ 3 && return repr(v)
    return "[ …, "*repr(last(v, 3))[2:end]
end
_repr(T::DataType) = "<:$(nameof(T))"

# Display, within containers, etc:
function Base.show(io::IO, trace::ControlTrace)
    str = "ControlTrace{$(_repr(typeof(trace.control)))}"
    if !isempty(trace)
        last_row = _last(trace.table)
        isempty(last_row) || (str *= ", ending $(_last(trace.table))")
    end
    print(io,  str)
end

# Display, standalone:
function Base.show(io::IO, ::MIME"text/plain", trace::ControlTrace)
    names = propertynames(trace)
    n = length(names) - 4
    names = names[1:n] # just the table column names
    str = "ControlTrace"
    isempty(trace) && (str *= " (empty)")
    str *= "\n  control: $(trace.control)"
    for name in  names
        column = getproperty(trace.table, name)
        str *= "\n  $name: "*_repr(column)
    end
    table_str = "("*join(map(name->"$name = …", names), ", ")*")"
    isnothing(trace.table) || (str *= "\n  table: $table_str")
    if isdefined(trace, :message) && !isempty(trace.message)
        str *= "\n  message: $(trace.message)"
    end
    str *= "\n  triggered_stop: $(trace.triggered_stop)"
    print(io, str)
end
