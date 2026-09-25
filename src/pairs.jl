"""
    context_pairs(sentence, window) -> Vector{Tuple{T,T}}

Every (center, context) pair within `window` positions inside one sentence, in
reading order. Pairs never cross a sentence boundary, and a word is never paired
with itself.

A sentence of `n` words yields `4n - 6` pairs when `window == 2` and `n >= 4`.

```jldoctest
julia> context_pairs(["the", "cat", "drinks", "milk"], 2)
10-element Vector{Tuple{String, String}}:
 ("the", "cat")
 ("the", "drinks")
 ("cat", "the")
 ("cat", "drinks")
 ("cat", "milk")
 ("drinks", "the")
 ("drinks", "cat")
 ("drinks", "milk")
 ("milk", "cat")
 ("milk", "drinks")
```
"""
function context_pairs(sentence::AbstractVector{T}, window::Integer) where {T}
    out = Tuple{T,T}[]
    n = length(sentence)
    for i in 1:n, j in max(1, i - window):min(n, i + window)
        j == i && continue
        push!(out, (sentence[i], sentence[j]))
    end
    out
end

"""
    corpus_pairs(sentences, vocab; window=2) -> Vector{Tuple{Int,Int}}

All training pairs of the corpus, as id pairs. Words missing from `vocab` are
skipped, and the words of a sentence keep their positions, so a dropped word
still separates its neighbours.
"""
function corpus_pairs(sentences, vocab::Vocabulary; window::Integer = 2)
    out = Tuple{Int,Int}[]
    for s in sentences
        ids = [get(vocab.index, w, 0) for w in s]
        for (c, o) in context_pairs(ids, window)
            (c == 0 || o == 0) && continue
            push!(out, (c, o))
        end
    end
    out
end

"""
    cooccurrence(pairs, V) -> Matrix{Int}
    cooccurrence(sentences, vocab; window=2) -> Matrix{Int}

The `V × V` table behind everything: entry `(i, j)` counts the pairs with word
`i` as the center and word `j` as the context.

The matrix is symmetric, because "`j` is within the window of `i`" is a symmetric
relation, and its total equals the number of training pairs.
"""
function cooccurrence(pairs::AbstractVector{Tuple{Int,Int}}, V::Integer)
    N = zeros(Int, V, V)
    for (c, o) in pairs
        N[c, o] += 1
    end
    N
end

cooccurrence(sentences, vocab::Vocabulary; window::Integer = 2) =
    cooccurrence(corpus_pairs(sentences, vocab; window), length(vocab))

"""
    row_totals(N) -> Vector{Int}

How often each word is the center of a pair: the row sums of the co-occurrence
matrix. Used by [`target_probability`](@ref) to work out how often a word is
drawn as a negative example.
"""
row_totals(N::AbstractMatrix{<:Integer}) = vec(sum(N, dims = 2))

"""
    pair_count(N, vocab, center, context) -> Int

How many pairs have `center` as the center and `context` as the context.
"""
pair_count(N::AbstractMatrix{<:Integer}, vocab::Vocabulary,
           center::AbstractString, context::AbstractString) =
    N[vocab[center], vocab[context]]
