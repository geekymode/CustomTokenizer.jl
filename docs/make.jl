using Documenter, CustomTokenizer
using Documenter: Remotes
using CairoMakie                       # activates the plotting methods for the examples

CairoMakie.activate!(type = "png")

DocMeta.setdocmeta!(CustomTokenizer, :DocTestSetup,
                    :(using CustomTokenizer); recursive = true)

makedocs(
    sitename = "CustomTokenizer.jl",
    repo     = Remotes.GitHub("geekymode", "CustomTokenizer.jl"),
    authors  = "word2vec-demo",
    modules  = [CustomTokenizer],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        assets     = String[],
    ),
    pages = [
        "Home"                => "index.md",
        "Getting started"     => "tutorial.md",
        "How the update works"=> "algorithm.md",
        "Stepping through it" => "stepping.md",
        "Reading the plots"   => "plots.md",
        "Scaling up"          => "scaling.md",
        "API reference"       => "api.md",
    ],
    checkdocs = :exports,
    doctest   = true,
    warnonly  = [:missing_docs],
)

deploydocs(
    repo       = "github.com/geekymode/CustomTokenizer.jl.git",
    devbranch  = "main",
    versions   = ["stable" => "v^", "v#.#", "dev" => "dev"],
)
