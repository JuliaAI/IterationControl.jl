module TablesExt

import IterationControl
import IterationControl.ControlTrace
import Tables

Tables.istable(::Type{<:ControlTrace}) = true
Tables.columnaccess(::Type{<:ControlTrace}) = true
Tables.columns(trace::ControlTrace) = Tables.columns(trace.table)
Tables.columnindex(trace::ControlTrace, idx::Symbol) = Tables.columnindex(trace.table)
Tables.columnindex(trace::ControlTrace, idx::AbstractString) =
    Tables.columnindex(trace, Symbol(idx))
Tables.schema(trace::ControlTrace) = Tables.schema(trace.table)
Tables.getcolumn(trace::ControlTrace, I) = Tables.getcolumn(trace.table, I)
Tables.subset(trace::ControlTrace, I; kwargs...) = Tables.subset(trace.table, I; kwargs...)

end # module
