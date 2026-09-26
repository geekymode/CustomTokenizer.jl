# Published vectors and analogies

Vectors trained here and vectors downloaded from elsewhere answer the same
questions through the same functions: [`Model`](@ref) and [`Embedding`](@ref)
are both [`AbstractEmbedding`](@ref).

## Loading

[`load_vectors`](@ref) reads the plain-text format that gensim, GloVe and
fastText all write — `word v1 v2 …`, one per line. A word2vec-style
`nwords dim` header is detected and skipped, so GloVe files (which have none)
load the same way.

```julia
glove = load_vectors("glove-50d-50k.vec"; max_words = 50_000, name = "GloVe 6B 50d")

nearest_neighbours(glove, "king", 5)
similarity(glove, "man", "woman")
```

`max_words` keeps only the first N entries; published files are sorted
most-frequent-first, so this trims the long tail rather than a random slice.

Going the other way, [`save_vectors`](@ref) exports a trained model:

```julia
save_vectors("ours.vec", embedding(model; name = "ours"))
```

To fetch GloVe in the first place:

```
python experiments/fetch_pretrained.py --words 50000 --dim 50
```

## Analogies

[`analogy`](@ref) answers "`a` is to `b` as `c` is to ?" — the words closest to
`b - a + c`, with `a`, `b` and `c` excluded, since they are always closest to
their own combination.

```julia
analogy(model, "man", "king", "woman")     # expect queen
analogy(model, "woman", "aunt", "man")     # expect uncle
```

[`analogy_accuracy`](@ref) runs a set of them, reporting words missing from the
vocabulary separately rather than scoring them as failures — coverage and
geometry are different problems.

```julia
acc = analogy_accuracy(model)              # uses ANALOGY_QUESTIONS
acc.accuracy, acc.hits, acc.asked, acc.skipped
```

**Analogies need a real corpus.** On the 22-word toy corpus they are
meaningless; on a million words of text8 they score zero; on the full 17M-word
text8 this package gets 7 of 12 at rank 1:

```
✓ man     → king   as woman  → ?   queen (0.65), mormaer, fortinbras
✓ man     → uncle  as woman  → ?   aunt (0.72), nieces, widower
✓ woman   → aunt   as man    → ?   uncle (0.66), romanoff, brother
✓ germany → berlin as japan  → ?   tokyo (0.67), beijing, osaka
✓ good    → better as bad    → ?   worse (0.55), faster, harder
  france  → paris  as italy  → ?   turin, bologna, verona     (right country, wrong city)
```

```
julia --project=. examples/analogies.jl ../text8 17000000
```

## An LLM's own input embeddings

A tokenizer has no vectors, so an analogy cannot be computed *in* one. The
model behind it does have vectors: the first thing a transformer does is look
up a row of its embedding table for each token id. Those rows can be extracted
and compared like any others.

```
julia --project=. experiments/extract_embeddings.jl                   # Qwen 2.5 0.5B
julia --project=. experiments/extract_embeddings.jl gpt2
```

The script downloads the checkpoint, reads the embedding tensor with a small
safetensors reader (no torch required, bfloat16 handled), keeps the rows whose
token is a whole word — `▁queen` becomes `queen`, while pieces like `ization`
are skipped — and writes a `.vec` file. Then:

```julia
llm = load_vectors("experiments/results/gemma-3-270m-embeddings.vec"; name = "Gemma 3 270M")

analogy(llm, "man", "king", "woman")
analogy_accuracy(llm)
neighbour_overlap(llm, glove, shared_vocabulary(llm, glove)[1:300]; k = 10)
```

`examples/compare_llm_embeddings.jl` does all of that for whichever vector
files it finds, and prints a per-question table of where the models disagree.

It is tempting to assume these rows must be poor word vectors, since they are
not trained to be any such thing — they feed a stack that adds context at every
layer. Measured, that assumption is wrong:

| | rank 1 | top 4 |
|---|---|---|
| ours (text8, 17M words) | 7/12 | 9/12 |
| GloVe (6B words, 50d) | 9/12 | 12/12 |
| **Qwen 2.5 0.5B input embeddings (896d)** | **11/12** | 11/12 |

The only one Qwen misses is `france → paris as italy → ?`, where it answers
`italian`. Its neighbour lists are clean too: `king → queen, kings, kingdom,
prince, monarch`.

What it does *not* share is geometry. Neighbour overlap against GloVe is 16%
and the similarity correlation is ≈0 — the space is arranged quite differently,
while still holding the same relations.

Which words are single tokens matters here, and differs by model:

| question | GPT-3 | GPT-4o | Gemma 4 | Gemma 2 |
|---|---|---|---|---|
| man/king/woman/queen | yes | yes | yes | yes |
| france/paris/italy/rome | no | no | yes | yes |
| germany/berlin/japan/tokyo | no | no | no | yes |

GPT-3 splits `paris` into `par|is` and `rome` into `r|ome`, so a model on that
tokenizer has no single row for either. `experiments/analogy_tokens.jl`
prints the full table.

## Comparing two models

Two embeddings live in unrelated coordinate systems, so the comparisons are
structural:

```julia
shared_vocabulary(ours, glove)                    # what can be compared at all
neighbour_overlap(ours, glove, words; k = 10)     # do the same words sit together?
similarity_agreement(ours, glove; pairs = 4000)   # do they rank pairs alike?
```

Running `examples/compare_pretrained.jl` on the full text8 model against GloVe
6B gives:

| | rank 1 | top 4 |
|---|---|---|
| ours (text8, 17M words, 100d, 3 epochs) | 7/12 | 9/12 |
| GloVe (6B words, 50d) | 9/12 | 12/12 |

```
neighbour overlap @10 over 500 shared words:  7.8%
similarity correlation over 4000 pairs:       0.395
```

That overlap looks alarming until you read the neighbours:

```
king   ours   fortinbras, pretender, mormaer, throne, eochaid
       GloVe  prince, queen, ii, emperor, son

war    ours   kursk, kharkov, wwii, taranaki, allied
       GloVe  occupation, invasion, wars, conflict, fighting
```

Every one of ours is a king or a word for one; `kursk` and `kharkov` are
battles. The vectors are coherent — they are simply Wikipedia-flavoured,
because text8 *is* Wikipedia, while GloVe also saw billions of words of news.

This is the distinction the measures are for. Undertraining shows up as
*incoherent* neighbours for common words. Ours are coherent but specific, which
is the corpus talking, not the training.
