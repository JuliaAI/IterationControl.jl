## DOC STRING GENERATION

function detailed_doc_string(M; header="", example="", body="")
    ret = "    $header"
    ret *= "\n\n"
    ret *= "A control strategy for use with the `IteratedModel` "*
        "wrapper, as in\n\n"*
    "    IteratedModel(model=NeuralNetworkRegressor(),\n"*
    "                  control=$example,\n"*
    "                  resampling=Holdout(fraction_train=0.7))"
    ret *= "\n\n"
    ret *= body
    return ret
end

_err_create_docs() = error(
"@create_docs syntax error. Usage: \n"*
    "@create_docs(ControlSubType, header=..., example=..., body=...)")

macro create_docs(M_ex, exs...)
    M_ex isa Symbol || _err_create_docs()
    h = ""
    e = ""
    b = ""
    for ex in exs
        ex.head == :(=) || _err_create_docs()
        ex.args[1] == :header &&  (h = ex.args[2])
        ex.args[1] == :example && (e = ex.args[2])
        ex.args[1] == :body &&    (b = ex.args[2])
    end
    esc(quote
        "$(IterationControl.detailed_doc_string($M_ex, header=$h, example=$e, body=$b))"
        function $M_ex end
        end)
end


## FOR EVALUATION OF FUNCTIONS PASSED TO LOGGING CONTROLS

_log_eval(f, model) = f(expose(model))
_log_eval(f::String, model) = f
