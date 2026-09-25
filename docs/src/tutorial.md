# Getting started

The pipeline has five stages, and each one is a function you can inspect.

## 1. Text to words

[`tokenize`](@ref) splits at full stops and keeps runs of letters, lowercased.
Punctuation, digits and capitals are discarded — see
[Scaling up](scaling.md) for how real models differ.

```@example tut
using CustomTokenizer
sentences = tokenize(TOY_TEXT)
sentences[1:2]
```

## 2. Words to ids

[`build_vocab`](@ref) counts the words and sorts them most-frequent-first, so
ids are stable and `1` is the commonest word.

```@example tut
vocab = build_vocab(sentences)
```

62 words of text, 22 distinct. From here on the model sees only ids.

```@example tut
vocab["cat"], vocab[3], vocab.counts[vocab["cat"]]
```

## 3. Two tables of numbers

[`Model`](@ref) allocates `W` (word vectors, kept) and `C` (context vectors,
discarded). `W` starts small and random, `C` starts at exactly zero.

```@example tut
model = Model(vocab; dim = 8, window = 2, negative = 3)
```

```@example tut
round.(wordvec(model, "cat"); digits = 4)
```

Every guess is 50/50 before training, because `C` is zero:

```@example tut
probability(model, "cat", "drinks")
```

## 4. Words to training pairs

[`corpus_pairs`](@ref) lists every (center, context) pair within the window.
Pairs never cross a sentence boundary.

```@example tut
pairs = corpus_pairs(sentences, vocab; window = 2)
length(pairs)
```

[`cooccurrence`](@ref) is the same information as a table, and it is all the
model will ever be told:

```@example tut
N = cooccurrence(sentences, vocab; window = 2)
sum(N), N == N'
```

King and queen have identical rows, which is why their vectors end up
identical:

```@example tut
ik, iq = vocab["king"], vocab["queen"]
[(vocab[j], N[ik, j], N[iq, j]) for j in 1:length(vocab) if N[ik, j] + N[iq, j] > 0]
```

## 5. Training

[`train!`](@ref) runs [`update_pair!`](@ref) over every pair, reshuffling each
epoch and drawing fresh negatives each time.

```@example tut
trainlog = train!(model, pairs; epochs = 300, snapshots = [0, 10, 300])
trainlog.losses[[1, 10, 100, 300]]
```

The loss starts at `4ln2 = 2.77`, the cost of guessing 50/50 on four questions,
and comes down.

## What came out

```@example tut
nearest_neighbours(model, "cat", 3)
```

```@example tut
[(a, b, round(similarity(model, a, b); digits = 2)) for (a, b) in
 (("king", "queen"), ("cat", "dog"), ("milk", "water"), ("cat", "king"))]
```

Nobody told the model that cats and dogs are animals. It found the grouping from
the co-occurrence table alone.

## Checking the result against the counts

Training settles where pulls and pushes cancel, at `N / (N + E)`.
[`target_probability`](@ref) computes that from the counts, and
[`calibration`](@ref) compares the whole trained model against it:

```@example tut
probs = sampling_probabilities(UniformNegatives(), length(vocab))
target_probability(N, row_totals(N), probs, vocab["cat"], vocab["drinks"], 3)
```

```@example tut
calibration(model, N, probs)
```

A correlation above 0.99 is the sense in which word2vec "is" a factorisation of
the co-occurrence table.
