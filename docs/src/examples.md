# Examples: how to load and run

Every script in `examples/` and `experiments/` is listed here with the command
that runs it and what it prints. The Julia ones need only the package; the
Python ones are for downloading real tokenizers and published vectors, and are
never needed to use the package itself.

Clone first, since the scripts live in the repository:

```
git clone https://github.com/geekymode/CustomTokenizer.jl
cd CustomTokenizer.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## The toy corpus, start to finish

```
julia --project=. examples/walkthrough.jl
```

Prints all seven stages: the sentences, the 22-word dictionary, the two tables,
the 176 training pairs, one traced update, the loss curve as numbers, the
similarities, the nearest neighbours, and a check of the trained probabilities
against what the counts predict.

## One update at a time

```
julia --project=. -i examples/stepping.jl      # -i keeps the REPL open
```

Traces the first updates, showing that `W` cannot move until some `C` column is
non-zero, then runs an epoch and reports how far the tables have grown. The
stepper is left live, so you can carry on:

```julia
julia> step!(stepper)
julia> progress(stepper)
julia> similarity(model, "king", "queen")
```

## All the figures

```
julia --project=docs examples/figures.jl
```

Writes nine PNGs into `examples/figures/`: both tables before and after an
update, the update anatomy, the loss curve, the similarity matrix, the
co-occurrence table, the two-dimensional map, and the evolution strip. This is
the fastest way to see what each plotting function produces.

## Analogies

Needs a real corpus — [text8](https://mattmahoney.net/dc/text8.zip) is the
usual one:

```
julia --project=. examples/analogies.jl ../text8 17000000
```

Trains on the slice (about six minutes for the full 17M words), asks twelve
questions, prints the neighbours of a few words, and saves the vectors to
`experiments/results/ours-text8.vec` for the comparisons below.

## Comparing with published vectors

```
pip install tiktoken tokenizers huggingface_hub gensim
python experiments/fetch_pretrained.py --words 50000 --dim 50
julia --project=. examples/compare_pretrained.jl
```

Downloads GloVe (6B tokens, 50 dimensions, 24 MB for the first 50k words) and
compares it with the model you trained: neighbour overlap, similarity
correlation, analogy scores, and the two sets of neighbours side by side.

## An LLM's own input embeddings

The first thing a transformer does is look up a row of its embedding table per
token. Those rows can be pulled out and compared like any other vectors:

```
python experiments/extract_llm_embeddings.py                       # Gemma 3 270M
python experiments/extract_llm_embeddings.py --repo Qwen/Qwen2.5-0.5B
julia --project=. examples/compare_llm_embeddings.jl
```

The extractor downloads the checkpoint (hundreds of MB to a couple of GB), reads
the embedding tensor with a small safetensors reader — no torch needed — keeps
the rows whose token is a whole word, and writes them as a `.vec` file.
`--self-test` checks the file parsing without downloading anything.

`compare_llm_embeddings.jl` uses whichever vector files it finds, so it works
with two of the three and gains the third later. It prints a per-question
table of where the models disagree:

```
    man     → king     as woman    → queen      ours: ✓             GloVe: ✓
    man     → uncle    as woman    → aunt       ours: ✓             GloVe: daughter
    france  → paris    as italy    → rome       ours: turin         GloVe: ✓
    big     → bigger   as small    → smaller    ours: ✓             GloVe: larger
```

## Real tokenizers

```
python experiments/tokenizer_zoo.py
julia --project=docs experiments/tokenizer_plots.jl
```

Downloads the tokenizer files for GPT-3, GPT-3.5/4, GPT-4o, Gemma 2, Gemma 4
and Qwen 2.5 — a few MB each, never the model weights — and measures tokens per
text, bytes per token, how digits and indentation and emoji are cut up, and
whether `decode(encode(x)) == x`. See [Tokenizer experiments](tokenizers.md)
for the results.

```
python experiments/analogy_tokens.py
```

Checks whether the analogy words survive as single tokens in each tokenizer,
which decides whether the model behind it has one vector for them at all.

## Does the tokenizer change the vectors?

```
python experiments/tokenize_corpus.py ../lee_background.cor
julia --project=docs experiments/subword_vs_word.jl
```

Tokenizes one corpus two ways — whole words and Gemma sub-word pieces — trains
the same model on each, and compares coverage, inflected-pair similarity and
neighbours.

## Which files each script writes

| script | writes |
|---|---|
| `examples/figures.jl` | `examples/figures/*.png` |
| `examples/analogies.jl` | `experiments/results/ours-text8.vec` |
| `experiments/fetch_pretrained.py` | `experiments/results/glove-*.vec` |
| `experiments/extract_llm_embeddings.py` | `experiments/results/*-embeddings.vec` |
| `experiments/tokenizer_zoo.py` | `experiments/results/tokenizers.json`, `*.tsv` |
| `experiments/tokenize_corpus.py` | `experiments/results/stream_*.txt`, `word_pieces.tsv` |

Vector files are gitignored — they are large and every script regenerates them.
