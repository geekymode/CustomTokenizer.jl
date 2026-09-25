# CustomTokenizer.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://geekymode.github.io/CustomTokenizer.jl/dev/)
[![Build](https://github.com/geekymode/CustomTokenizer.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/geekymode/CustomTokenizer.jl/actions/workflows/documentation.yml)

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

Loading it is what activates `plot_tables` and the rest — no other
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

## Examples

Every script, with the command that runs it, is catalogued at
<https://geekymode.github.io/CustomTokenizer.jl/dev/examples/>. In short:

```
julia --project=. examples/walkthrough.jl            # the toy corpus, start to finish
julia --project=. -i examples/stepping.jl            # one update at a time
julia --project=docs examples/figures.jl             # write all nine figures
julia --project=. examples/analogies.jl ../text8 17000000   # train, then ask analogies

python experiments/tokenizer_zoo.py                  # GPT-3/4o/Gemma/Qwen side by side
python experiments/analogy_tokens.py                 # are the analogy words single tokens?
python experiments/fetch_pretrained.py               # download GloVe
python experiments/extract_llm_embeddings.py         # pull an LLM's embedding table
julia --project=. examples/compare_llm_embeddings.jl # compare all of them
```

## Documentation

**<https://geekymode.github.io/CustomTokenizer.jl/dev/>**, rebuilt on every push.
To build it locally:

```
cd docs && julia --project=. make.jl     # writes docs/build/index.html
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
