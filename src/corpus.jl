"""
    TOY_TEXT

Twelve sentences about cats, dogs, kings and queens: 62 words, 22 of them
distinct. Small enough that every table in this package fits on a screen, and
structured enough that training has something to find — the cat/dog sentences
and the king/queen sentences share no content words.
"""
const TOY_TEXT = """
The cat drinks milk. The dog drinks water.
The cat chases the mouse. The dog chases the cat.
A cat sleeps on the sofa. A dog sleeps on the floor.
The king rules the kingdom. The queen rules the kingdom.
The king wears a crown. The queen wears a crown.
The king lives in a castle. The queen lives in a castle.
"""

"""
    tokenize(text; delimiter='.', pattern=r"[A-Za-z]+", lower=true) -> Vector{Vector{String}}

Split `text` into sentences at `delimiter`, then each sentence into the runs of
characters matching `pattern`. Empty sentences are dropped.

This is deliberately the simplest tokenizer that works: punctuation, digits and
capitalisation are discarded, and a word that was never seen has no
representation at all. Real models use reversible sub-word tokenizers instead;
the difference is discussed in the documentation.

```jldoctest
julia> tokenize("The cat drinks milk. The dog drinks water.")
2-element Vector{Vector{String}}:
 ["the", "cat", "drinks", "milk"]
 ["the", "dog", "drinks", "water"]
```
"""
function tokenize(text::AbstractString; delimiter::Char = '.',
                  pattern::Regex = r"[A-Za-z]+", lower::Bool = true)
    out = Vector{Vector{String}}()
    for sentence in split(text, delimiter)
        words = [String(m.match) for m in eachmatch(pattern, sentence)]
        lower && (words = lowercase.(words))
        isempty(words) || push!(out, words)
    end
    out
end

"""
    subsample_probabilities(vocab; threshold=1e-3) -> Vector{Float64}

Probability of *keeping* each occurrence of each word, following the rule used by
the original word2vec: a word occurring with frequency `f` is kept with
probability `min(1, (sqrt(f/t) + 1) * t/f)`.

Very frequent words ("the") are kept only sometimes, so they stop dominating the
training pairs. On a corpus as small as [`TOY_TEXT`](@ref) nothing is dropped,
which is exactly why "the" dominates its first epochs.
"""
function subsample_probabilities(vocab::Vocabulary; threshold::Real = 1e-3)
    total = sum(vocab.counts)
    [min(1.0, (sqrt(c / (threshold * total)) + 1) * threshold * total / c) for c in vocab.counts]
end

"""
    subsample(sentences, vocab, keep, rng) -> Vector{Vector{Int}}

Turn sentences of words into sentences of ids, dropping each occurrence with the
probability given by [`subsample_probabilities`](@ref). Words outside the
vocabulary are removed.
"""
function subsample(sentences, vocab::Vocabulary, keep::AbstractVector{<:Real},
                   rng::AbstractRNG = Random.default_rng())
    out = Vector{Vector{Int}}()
    for s in sentences
        ids = Int[]
        for w in s
            i = get(vocab.index, w, 0)
            i == 0 && continue
            rand(rng) < keep[i] && push!(ids, i)
        end
        isempty(ids) || push!(out, ids)
    end
    out
end
