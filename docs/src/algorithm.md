# How the update works

One call to [`update_pair!`](@ref) is the entire algorithm. This page is the
arithmetic behind it.

## The question the model answers

For a center word ``i`` and another word ``j``, the model states a probability:

```math
p_{ij} = \sigma\!\left(w_i \cdot c_j\right), \qquad \sigma(s) = \frac{1}{1+e^{-s}}
```

where ``w_i`` is a column of `W` and ``c_j`` a column of `C`. Doing this for
every combination at once is a matrix product, ``S = W^\top C``: the two tables
are the **two factors** of one ``V \times V`` score matrix.

```@example alg
using CustomTokenizer
sentences = tokenize(TOY_TEXT); vocab = build_vocab(sentences)
model = Model(vocab)
size(model.W), size(model.C), size(model.W' * model.C)
```

Because a ``22 \times 22`` table has to pass through only 8 numbers per word,
words with similar rows are forced to share similar vectors. That pressure is
where the similarity comes from.

## One update, four questions

An update takes the real context word plus `negative` random words, and labels
them: ``y = 1`` for the real neighbour, ``y = 0`` for the random ones.

```@example alg
info = update_pair!(model, "cat", "drinks", ["on", "wears", "queen"], 0.5)
```

## The loss

The loss is the **binary cross-entropy** of those four answers — the negative
log of the probability the model gave to the correct ones:

```math
L = -\sum_j \Bigl[\, y_j \log p_j + (1-y_j) \log (1-p_j) \Bigr]
```

The label ``y`` is not a new quantity: it is a switch that writes the two cases
as one expression. At 50/50 each question costs ``\ln 2``, so an untrained update
costs ``4\ln 2``:

```@example alg
info.loss, 4log(2)
```

## Its gradient

With ``s_j = w_i \cdot c_j``:

1. ``\sigma'(s) = \sigma(s)(1-\sigma(s)) = p(1-p)``
2. therefore ``\partial L/\partial s_j = p_j - y_j`` — the ``1/p`` from the
   logarithm cancels the ``p(1-p)`` from the sigmoid, and only the error
   survives
3. the score is a dot product, so ``\partial s_j/\partial c_j = w_i`` and
   ``\partial s_j/\partial w_i = c_j``
4. chaining: ``\partial L/\partial c_j = (p_j - y_j) w_i`` and
   ``\partial L/\partial w_i = \sum_j (p_j - y_j) c_j``

Descent flips the sign, giving `g = (want - p) * lr` and

```math
c_j \leftarrow c_j + g_j\, w_i, \qquad w_i \leftarrow w_i + \sum_j g_j\, c_j
```

Every other column appears nowhere in ``L``, so its derivative is exactly zero.
That is the formal reason only a handful of columns move.

## Why cross-entropy and not squared error

Paired with the sigmoid, cross-entropy leaves the clean error `p - y`. Squared
error would carry a factor ``p(1-p)``, which is near zero exactly where the model
is confidently wrong:

```@example alg
p = 0.01                      # want 1: as wrong as it gets
cross_entropy = abs(p - 1)
squared_error = 2 * abs(p - 1) * p * (1 - p)
(cross_entropy, squared_error, cross_entropy / squared_error)
```

Fifty times weaker, in the case that most needs fixing.

## The update is rank one

Every changed column of `C` is a multiple of the *same* vector ``w_i``, so
``\Delta C = w_i \tilde g^\top`` is an outer product:

```@example alg
m2 = Model(vocab)
pairs = corpus_pairs(sentences, vocab; window = 2)
train!(m2, pairs; epochs = 5)                      # warm up so both tables move

W0, C0 = copy(m2.W), copy(m2.C)
u = update_pair!(m2, "queen", "the", ["floor", "on", "crown"], 0.05)

moved_W = [vocab[j] for j in 1:length(vocab) if m2.W[:, j] != W0[:, j]]
moved_C = [vocab[j] for j in 1:length(vocab) if m2.C[:, j] != C0[:, j]]
(moved_W, moved_C)
```

```@example alg
w = W0[:, vocab["queen"]]
ΔC_the = m2.C[:, vocab["the"]] - C0[:, vocab["the"]]
ΔC_the ≈ u.errors[1] .* w
```

## Where training settles

Per epoch a pair receives ``N_{ij}`` pulls and, on average,
``k\,(R_i - N_{ij})\,P(j)`` pushes, where ``R_i`` is the row total and ``P(j)``
the chance of drawing ``j`` as a negative. They cancel at

```math
p^\star = \frac{N_{ij}}{N_{ij} + E_{ij}}
\qquad\Longleftrightarrow\qquad
w_i \cdot c_j \approx \log \frac{N_{ij}}{E_{ij}}
```

so the trained score is a log-ratio: not "how often do these co-occur" but "how
much more often than chance". [`target_probability`](@ref) evaluates it:

```@example alg
N = cooccurrence(sentences, vocab; window = 2)
probs = sampling_probabilities(UniformNegatives(), length(vocab))
[(b, round(target_probability(N, row_totals(N), probs, vocab["cat"], vocab[b], 3); digits = 2))
 for b in ("the", "chases", "drinks", "crown")]
```

With only 22 words a real neighbour is drawn as a "random" word about as often as
it appears as a pair, which is why nothing reaches 1. On a real corpus the
pushes spread over thousands of words and content pairs settle much higher.
