using Documenter
using DocumenterCodeBlocks
using IterationControl
import EarlyStopping

const  REPO = Remotes.GitHub("JuliaAI", "IterationControl.jl")

makedocs(
    modules=[IterationControl, EarlyStopping],
    format=Documenter.HTML(
        prettyurls = true,
        collapselevel = 1,
    ),
    plugins = [CodeBlocks(),],
    pages=[
        "Overview" => "index.md",
        "Controls" => "controls.md",
        "Adding New Controls" => "adding_new_controls.md",
        "Demos" => [
            "Particle Tracker" => "notebooks/particle/notebook.md",
            "Tree Booster" => "notebooks/tree_booster/notebook.md",
            "Anteaters" => "notebooks/anteaters/notebook.md",
            ],
    ],
    sitename="IterationControl.jl",
    warnonly = [:cross_references, :missing_docs],
    repo = Remotes.GitHub("JuliaAI", "IterationControl.jl"),
)

deploydocs(
    devbranch="dev",
    push_preview=false,
    repo="github.com/JuliaAI/IterationControl.jl.git",
)
