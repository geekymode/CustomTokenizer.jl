"""
    Model

The two tables the algorithm trains, plus the vocabulary that indexes them.

* `W` — `dim × V`, the **word vectors**. Column `i` describes word `i` when it is
  the *center* of a pair. This is the table that is kept.
* `C` — `dim × V`, the **context vectors**. Column `j` describes word `j` when it
  is a *neighbour*. Discarded after training.

`W` starts as small random numbers and `C` starts at exactly zero. Only one of
the two needs random values: the first updates copy scaled multiples of `W`
columns into `C`, after which both tables move. If both started at zero nothing
would ever change.

    Model(vocab; dim=8, window=2, negative=3, rng=Xoshiro(7))

The default `rng` reproduces the numbers used throughout the documentation.
"""
mutable struct Model <: AbstractEmbedding
    vocab::Vocabulary
    W::Matrix{Float64}
    C::Matrix{Float64}
    window::Int
    negative::Int
end

function Model(vocab::Vocabulary; dim::Integer = 8, window::Integer = 2,
               negative::Integer = 3, rng::AbstractRNG = Xoshiro(7))
    V = length(vocab)
    W = (rand(rng, dim, V) .- 0.5) ./ dim
    C = zeros(dim, V)
    Model(vocab, W, C, window, negative)
end

Base.size(m::Model) = size(m.W)

function Base.show(io::IO, m::Model)
    print(io, "Model(", size(m.W, 1), " × ", size(m.W, 2), ", window ", m.window,
          ", ", m.negative, " negatives)")
end

function Base.show(io::IO, ::MIME"text/plain", m::Model)
    d, V = size(m.W)
    println(io, "Model: ", d, " numbers per word, ", V, " words")
    @printf(io, "  window %d · %d negative samples per pair\n", m.window, m.negative)
    @printf(io, "  |W| largest entry %.4f, |C| largest entry %.4f\n",
            maximum(abs, m.W), maximum(abs, m.C))
end

_id(m::AbstractEmbedding, w::AbstractString) = m.vocab[w]
_id(::AbstractEmbedding, i::Integer) = Int(i)

"""
    wordvec(model, word) -> AbstractVector

Column of `W` for `word` (a string or an id). This is the vector that is kept
after training.
"""
wordvec(m::AbstractEmbedding, w) = view(m.W, :, _id(m, w))

"""
    ctxvec(model, word) -> AbstractVector

Column of `C` for `word`: how the word behaves as somebody else's neighbour.
"""
ctxvec(m::Model, w) = view(m.C, :, _id(m, w))

"""
    score(model, center, context) -> Float64

The dot product `W[:, center] ⋅ C[:, context]`, before the sigmoid. Training
drives this quantity toward `log(N / E)`: the log-ratio of how often the pair
occurs to how often chance alone would produce it.
"""
score(m::Model, center, context) = dot(wordvec(m, center), ctxvec(m, context))

sigmoid(x::Real) = 1 / (1 + exp(-x))

"""
    probability(model, center, context) -> Float64

`σ(score)`: the model's estimate that `context` appears near `center`. Every
guess is exactly 0.5 before training, because `C` is zero.
"""
probability(m::Model, center, context) = sigmoid(score(m, center, context))

"""
    cosine_similarity(u, v) -> Float64

Cosine of the angle between two vectors: `+1` same direction, `0` unrelated,
`-1` opposite.
"""
cosine_similarity(u::AbstractVector, v::AbstractVector) = dot(u, v) / (norm(u) * norm(v))

"""
    similarity(model, a, b) -> Float64

Cosine similarity of two *word* vectors (columns of `W`). This is the number
that says cat is like dog.
"""
similarity(m::AbstractEmbedding, a, b) = cosine_similarity(wordvec(m, a), wordvec(m, b))

"""
    nearest_neighbours(model, word, k=5) -> Vector{Tuple{String,Float64}}

The `k` words whose vectors point most like `word`'s, closest first, excluding
`word` itself.
"""
function nearest_neighbours(m::AbstractEmbedding, word, k::Integer = 5)
    i = _id(m, word)
    V = length(m.vocab)
    sims = [j == i ? -Inf : similarity(m, i, j) for j in 1:V]
    [(m.vocab[j], sims[j]) for j in partialsortperm(sims, 1:min(k, V - 1); rev = true)]
end

"""
    analogy(model, a, b, c; k=3) -> Vector{Tuple{String,Float64}}

Answer "`a` is to `b` as `c` is to ?" by looking for the words closest to
`b - a + c`, excluding the three inputs. A toy corpus rarely has enough evidence
for this to work; it is here because the question always comes up.
"""
function analogy(m::AbstractEmbedding, a, b, c; k::Integer = 3)
    ia, ib, ic = _id(m, a), _id(m, b), _id(m, c)
    target = wordvec(m, ib) .- wordvec(m, ia) .+ wordvec(m, ic)
    V = length(m.vocab)
    sims = [j in (ia, ib, ic) ? -Inf : cosine_similarity(target, wordvec(m, j)) for j in 1:V]
    [(m.vocab[j], sims[j]) for j in partialsortperm(sims, 1:min(k, V - 3); rev = true)]
end

"""
    column_norms(M) -> Vector{Float64}

Length of every column of a table. Useful for watching `C` grow from zero, and
for seeing that frequent words grow first.
"""
column_norms(M::AbstractMatrix) = [norm(view(M, :, j)) for j in axes(M, 2)]
