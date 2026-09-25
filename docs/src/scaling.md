# Scaling up

The defaults suit 22 words. This page is what changes when the corpus is real,
and which knobs exist for it.

## The pieces a real corpus needs

| problem | what the package offers |
|---|---|
| most word types occur once or twice | `build_vocab(sentences; min_count = 5)` |
| a few words are a quarter of the corpus | [`subsample_probabilities`](@ref), [`subsample`](@ref) |
| uniform negatives hit only rare words | [`UnigramNegatives`](@ref) |
| one pass is enough, not 300 | `train!(...; epochs = 5)` |

## A two-cluster example

A synthetic corpus large enough to need those settings, and small enough to run
in the documentation build:

```@example scale
using CustomTokenizer, Random

text = repeat("red apple tastes sweet. green apple tastes sour. " *
              "blue car drives fast. black car drives slow. ", 200)
sentences = tokenize(text)
vocab = build_vocab(sentences; min_count = 5)
length(vocab), sum(vocab.counts)
```

```@example scale
pairs = corpus_pairs(sentences, vocab; window = 5)
model = Model(vocab; dim = 24, window = 5, negative = 5, rng = Xoshiro(3))
sampler = UnigramNegatives(vocab; table_size = 50_000)
trainlog = train!(model, pairs; epochs = 20, sampler = sampler, rng = Xoshiro(3))
(length(pairs), round(trainlog.losses[1]; digits = 3), round(trainlog.losses[end]; digits = 3))
```

The two clusters share no words, so they separate:

```@example scale
(round(similarity(model, "apple", "tastes"); digits = 2),
 round(similarity(model, "apple", "car"); digits = 2))
```

## Negative sampling changes character

With a uniform draw every word is equally likely; with
[`UnigramNegatives`](@ref) the draw follows `count^0.75`, so pushes land on words
the model actually meets:

```@example scale
uni  = sampling_probabilities(UniformNegatives(), length(vocab))
prop = sampling_probabilities(sampler, length(vocab))
[(vocab[i], vocab.counts[i], round(uni[i]; digits = 4), round(prop[i]; digits = 4))
 for i in (1, 2, length(vocab))]
```

## Subsampling the frequent words

```@example scale
keep = subsample_probabilities(vocab; threshold = 1e-3)
[(vocab[i], vocab.counts[i], round(keep[i]; digits = 3)) for i in (1, 2, length(vocab))]
```

Each occurrence of a very frequent word is kept only with that probability, which
stops it consuming the run:

```@example scale
ids = subsample(sentences, vocab, keep, Xoshiro(1))
(sum(length, sentences), sum(length, ids))
```

## Where the probabilities settle, at two scales

The balance point `N/(N+E)` depends on how often the *partner* word is drawn as
a negative. In a small vocabulary that is often; in a large one it is rare, so
content pairs settle much higher:

```@example scale
N = cooccurrence(sentences, vocab; window = 5)
totals = row_totals(N)
[(a, b, round(target_probability(N, totals, prop, vocab[a], vocab[b], 5); digits = 2))
 for (a, b) in (("apple", "tastes"), ("apple", "red"), ("car", "drives"))]
```

## Things that do not change

The update rule, the loss, the rank-one structure, and the fact that the trained
score approximates `log(N/E)`. Only the machinery around them has to grow.

## A note on the tokenizer

[`tokenize`](@ref) is the simplest thing that works: lowercase, letters only,
split at full stops. Real models use reversible sub-word tokenizers — a fixed
vocabulary of word pieces, digits split apart, whitespace preserved and byte
fallback so nothing is ever out of vocabulary. The embedding table is the same
object as `W` in this package, with a few hundred thousand columns instead of 22.
