# CustomTokenizer.jl

Skip-gram word2vec with negative sampling, written to be **read** rather than to
be fast. Every step of the algorithm is a named function you can call on its own,
print, plot and test.

The package grew out of a twelve-sentence teaching script. That corpus ships as
`TOY_TEXT`, and the defaults match it, so the entire state of a training run —
both tables, every training pair — fits on one screen.

```julia
using CustomTokenizer

sentences = tokenize(TOY_TEXT)
vocab     = build_vocab(sentences)          # 22 words, 62 tokens
model     = Model(vocab; dim = 8, window = 2, negative = 3)
pairs     = corpus_pairs(sentences, vocab; window = 2)   # 176 pairs

train!(model, pairs; epochs = 300)

similarity(model, "king", "queen")    # ≈ 1.00  (identical neighbours)
similarity(model, "cat", "dog")       # ≈ 0.75
similarity(model, "cat", "king")      # ≈ 0.2   (different kinds of sentence)
nearest_neighbours(model, "cat", 3)   # dog, mouse, milk
```

Nobody tells the model that cats and dogs are animals. It finds the grouping from
the co-occurrence counts alone.

## What makes it useful for learning

**One update is one function call, and it reports what it did.**

```julia
info = update_pair!(model, "cat", "drinks", ["on", "wears", "queen"], 0.5)
```

```
update: center cat · context drinks · negatives on, wears, queen · lr 0.5000
  word        want       p         g
  drinks        1    0.50000  +0.25000
  on            0    0.50000  -0.25000
  wears         0    0.50000  -0.25000
  queen         0    0.50000  -0.25000
  loss 2.77259 · |ΔW[:, cat]| 0.000e+00
```

That last number is not a bug: on the first update `C` is still zero, so the
center cannot move yet. The package makes facts like this checkable rather than
assertable — it is in the test suite.

**Training can be stepped.**

```julia
s = TrainStepper(model, pairs)
step!(s)              # one update
run_updates!(s, 25)   # twenty-five
finish_epoch!(s)      # to the end of the epoch
progress(s)           # (update = 202, epoch = 2, pair = 26, of = 176)
reset!(s)             # replay from the beginning, exactly
```

**Every claim has a number behind it.** Where a pair's probability settles is
predicted from the counts alone, and the trained model is checked against it:

```julia
N     = cooccurrence(sentences, vocab; window = 2)
probs = sampling_probabilities(UniformNegatives(), length(vocab))

target_probability(N, row_totals(N), probs, vocab["cat"], vocab["drinks"], 3)  # 0.40
calibration(model, N, probs)    # (correlation = 0.99, mean_absolute_error = 0.016, n = 462)
```

**Plots are one call each**, and live in a package extension — the core package
depends on nothing beyond the standard library.

```julia
using CairoMakie                    # activates the plotting methods

plot_tables(model)                  # W above C, every value labelled
plot_tables(model; highlight = info)# with the columns this update touched outlined
plot_update(model, info)            # the four questions and the rank-one change
plot_loss(trainlog)                 # loss per epoch against the coin-flip line
plot_similarity_matrix(model)       # every word against every word
plot_cooccurrence(N, vocab)         # the counts the model learns from
plot_embedding_map(model)           # the vectors in two dimensions
plot_evolution(model, trainlog.snapshots)   # how they got there
```

## Installation

```julia
using Pkg
Pkg.develop(path = "path/to/CustomTokenizer")
Pkg.add("CairoMakie")     # optional, for the figures
```

## Documentation

```
cd docs && julia --project=. make.jl     # builds docs/build/index.html
```

Pages: getting started, how the update works (loss, gradient, the rank-one
structure, where training settles), stepping through it, reading the plots,
scaling up to a real corpus, and the full API reference.

## Tests

```julia
using Pkg; Pkg.test("CustomTokenizer")
```

174 tests, covering the tokenizer, the vocabulary, pair generation, both negative
samplers, the documented numbers of the worked example, the invariants of one
update (only five columns move; each `ΔC` is a multiple of the center's column;
a repeated negative is pushed twice), a **finite-difference check of the
gradient**, reproducibility, the stepper against `train!`, the analysis helpers,
and that every plot renders.

## Layout

| file | contents |
|---|---|
| `src/corpus.jl` | `TOY_TEXT`, `tokenize`, subsampling |
| `src/vocab.jl` | `Vocabulary`, `build_vocab` |
| `src/pairs.jl` | `context_pairs`, `corpus_pairs`, `cooccurrence` |
| `src/model.jl` | `Model`, `wordvec`, `score`, `probability`, `similarity`, `nearest_neighbours` |
| `src/sampling.jl` | `UniformNegatives`, `UnigramNegatives` |
| `src/train.jl` | `update_pair!`, `train!`, `learning_rate`, `TrainLog` |
| `src/stepping.jl` | `TrainStepper` and friends |
| `src/analysis.jl` | targets, calibration, similarity matrix, PCA, ranking churn |
| `ext/` | the Makie plotting extension |

## A note on the name `update_pair!`

The teaching script calls it `update!`. Makie exports a function of that name, so
the package uses `update_pair!` to keep `using CustomTokenizer, CairoMakie`
free of clashes.
