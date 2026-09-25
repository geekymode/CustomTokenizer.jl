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

```julia
using Pkg
Pkg.develop(path = "path/to/CustomTokenizer")
Pkg.add("CairoMakie")      # optional, for the figures
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
