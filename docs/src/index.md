# CustomTokenizer.jl

Skip-gram word2vec with negative sampling, written to be read rather than to be
fast. Every step of the algorithm is a named function you can call on its own,
print, plot and test.

The package grew out of a twelve-sentence teaching script. That corpus ships as
[`TOY_TEXT`](@ref) and the defaults match it, so the whole state of a run fits on
one screen:

```@example home
using CustomTokenizer

sentences = tokenize(TOY_TEXT)
vocab     = build_vocab(sentences)
```

Nothing here is a black box. The two tables the algorithm trains are ordinary
matrices you can index:

```@example home
model = Model(vocab; dim = 8, window = 2, negative = 3)
round.(wordvec(model, "cat"); digits = 4)
```

and one training step is one function call that tells you what it did:

```@example home
update_pair!(model, "cat", "drinks", ["on", "wears", "queen"], 0.5)
```

After the full run, words used in the same kind of sentence end up pointing the
same way:

```@example home
pairs = corpus_pairs(sentences, vocab; window = 2)
train!(model, pairs; epochs = 300)

[(a, b, round(similarity(model, a, b); digits = 2)) for (a, b) in
 (("king", "queen"), ("cat", "dog"), ("milk", "water"), ("cat", "king"))]
```

## What is in the package

| area | functions |
|---|---|
| corpus | [`tokenize`](@ref), [`TOY_TEXT`](@ref), [`subsample_probabilities`](@ref), [`subsample`](@ref) |
| vocabulary | [`Vocabulary`](@ref), [`build_vocab`](@ref), [`vocab_size`](@ref) |
| pairs | [`context_pairs`](@ref), [`corpus_pairs`](@ref), [`cooccurrence`](@ref), [`row_totals`](@ref) |
| model | [`Model`](@ref), [`wordvec`](@ref), [`ctxvec`](@ref), [`score`](@ref), [`probability`](@ref), [`similarity`](@ref), [`nearest_neighbours`](@ref), [`analogy`](@ref) |
| negatives | [`UniformNegatives`](@ref), [`UnigramNegatives`](@ref), [`sample_negatives`](@ref), [`sampling_probabilities`](@ref) |
| training | [`update_pair!`](@ref), [`train!`](@ref), [`learning_rate`](@ref), [`TrainLog`](@ref), [`evaluate_loss`](@ref) |
| stepping | [`TrainStepper`](@ref), [`step!`](@ref), [`run_updates!`](@ref), [`finish_epoch!`](@ref), [`reset!`](@ref), [`progress`](@ref) |
| analysis | [`target_probability`](@ref), [`target_matrix`](@ref), [`calibration`](@ref), [`similarity_matrix`](@ref), [`pca2`](@ref), [`neighbour_ranking`](@ref), [`ranking_churn`](@ref) |
| embeddings | [`Embedding`](@ref), [`embedding`](@ref), [`load_vectors`](@ref), [`save_vectors`](@ref) |
| comparing | [`shared_vocabulary`](@ref), [`neighbour_overlap`](@ref), [`similarity_agreement`](@ref), [`analogy_accuracy`](@ref) |
| plots | [`plot_tables`](@ref), [`plot_update`](@ref), [`plot_loss`](@ref), [`plot_similarity_matrix`](@ref), [`plot_cooccurrence`](@ref), [`plot_embedding_map`](@ref), [`plot_evolution`](@ref) |

Plotting lives in a package extension: the core package has no dependencies
beyond the standard library, and the plotting methods appear as soon as you load
a Makie backend.

## Installation

This package is **not in the General registry**, so `Pkg.add("CustomTokenizer")`
will not find it. Install it from the repository URL instead:

```julia
using Pkg
Pkg.add(url = "https://github.com/geekymode/CustomTokenizer.jl")
```

or, in the Pkg REPL mode that `]` opens:

```
pkg> add https://github.com/geekymode/CustomTokenizer.jl
```

That tracks the default branch. To pin a branch or a tag, add `rev`:

```julia
Pkg.add(url = "https://github.com/geekymode/CustomTokenizer.jl", rev = "main")
```

Julia 1.9 or newer is required; the package is developed on 1.13. Updating
later is `Pkg.update("CustomTokenizer")`; removing it is
`Pkg.rm("CustomTokenizer")`.

### Plots

The figures need a Makie backend, which is a separate install because the core
package deliberately depends on nothing beyond the standard library:

```julia
Pkg.add("CairoMakie")     # or GLMakie for an interactive window
```

Loading it is what activates [`plot_tables`](@ref) and the rest — no other
change is needed.

### Working on the package itself

To edit the source, clone it and work in its own environment:

```
git clone https://github.com/geekymode/CustomTokenizer.jl
cd CustomTokenizer.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

`Pkg.develop(path = ".")` from another environment points that environment at
your clone, so edits take effect without reinstalling.

### The experiment scripts

`experiments/` compares this package with real LLM tokenizers and with
published vectors. Those scripts are Python, and are not needed to use the
package:

```
pip install tiktoken tokenizers huggingface_hub gensim
```

They download tokenizer files and word vectors (tens of MB), never model
weights.

### Checking that it works

```julia
using CustomTokenizer

sentences = tokenize(TOY_TEXT)
vocab     = build_vocab(sentences)
model     = Model(vocab)
train!(model, corpus_pairs(sentences, vocab; window = 2); epochs = 300)

similarity(model, "king", "queen")     # ≈ 1.0 if everything is working
```

## Where to go next

* [Getting started](tutorial.md) — the whole pipeline in one page.
* [How the update works](algorithm.md) — the loss, its gradient, and why the
  update is rank one.
* [Stepping through it](stepping.md) — run one update at a time and watch the
  tables change.
* [Reading the plots](plots.md) — what each figure shows.
* [Scaling up](scaling.md) — what changes on a real corpus.
* [Published vectors and analogies](pretrained.md) — load GloVe, ask
  "man is to king as woman is to ?", and compare a model you trained against
  one you downloaded.
* [Tokenizer experiments](tokenizers.md) — how GPT-3, GPT-4o and Gemma 4 cut
  the same text up, and whether the choice changes the vectors.
