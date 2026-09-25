"""
    shared_vocabulary(a, b) -> Vector{String}

Words both embeddings know, in the order of the first. Every comparison below
is restricted to these, since nothing else can be compared.
"""
shared_vocabulary(a::AbstractEmbedding, b::AbstractEmbedding) =
    [w for w in a.vocab.words if haskey(b.vocab, w)]

"""
    neighbour_overlap(a, b, words; k=10) -> NamedTuple

For each word, how many of its `k` nearest neighbours in `a` are also among its
`k` nearest in `b`, as a fraction. Returns `(mean, per_word)`.

This is the most direct "do these two models agree?" measure: it ignores the
coordinate systems, which are arbitrary and unrelated, and compares only the
neighbourhood structure.
"""
function neighbour_overlap(a::AbstractEmbedding, b::AbstractEmbedding,
                           words::AbstractVector{<:AbstractString}; k::Integer = 10)
    per = Dict{String,Float64}()
    for w in words
        (haskey(a.vocab, w) && haskey(b.vocab, w)) || continue
        na = Set(first.(nearest_neighbours(a, w, k)))
        nb = Set(first.(nearest_neighbours(b, w, k)))
        per[w] = length(intersect(na, nb)) / k
    end
    (mean = isempty(per) ? 0.0 : mean(values(per)), per_word = per)
end

"""
    similarity_agreement(a, b; pairs=2000, rng=Xoshiro(7)) -> NamedTuple

Correlation between the two embeddings' cosine similarities, over randomly
drawn pairs of shared words. `(correlation, n)`.

A high correlation means the two models rank word pairs the same way even
though their vectors live in unrelated coordinate systems.
"""
function similarity_agreement(a::AbstractEmbedding, b::AbstractEmbedding;
                              pairs::Integer = 2000, rng::AbstractRNG = Xoshiro(7))
    shared = shared_vocabulary(a, b)
    length(shared) < 3 && return (correlation = NaN, n = 0)
    xs = Float64[]
    ys = Float64[]
    for _ in 1:pairs
        u, v = rand(rng, shared), rand(rng, shared)
        u == v && continue
        push!(xs, similarity(a, u, v))
        push!(ys, similarity(b, u, v))
    end
    (correlation = cor(xs, ys), n = length(xs))
end

"""
    ANALOGY_QUESTIONS

A small set of `(a, b, c, expected)` questions — "a is to b as c is to
expected" — mixing semantic (capitals, family) and syntactic (plurals, tenses)
relations. Enough to compare two models; not a benchmark.
"""
const ANALOGY_QUESTIONS = [
    ("man", "king", "woman", "queen"),
    ("woman", "queen", "man", "king"),
    ("man", "uncle", "woman", "aunt"),
    ("woman", "aunt", "man", "uncle"),
    ("boy", "son", "girl", "daughter"),
    ("france", "paris", "italy", "rome"),
    ("germany", "berlin", "japan", "tokyo"),
    ("england", "london", "france", "paris"),
    ("good", "better", "bad", "worse"),
    ("big", "bigger", "small", "smaller"),
    ("walk", "walking", "swim", "swimming"),
    ("day", "days", "year", "years"),
]

"""
    analogy_accuracy(emb, questions=ANALOGY_QUESTIONS; k=1) -> NamedTuple

Share of questions whose expected answer appears in the top `k` results of
[`analogy`](@ref). Questions with a word outside the vocabulary are reported
separately rather than counted as failures, since they test coverage, not
geometry.

Returns `(accuracy, hits, asked, skipped, results)`.
"""
function analogy_accuracy(e::AbstractEmbedding,
                          questions = ANALOGY_QUESTIONS; k::Integer = 1)
    hits = 0
    asked = 0
    skipped = String[]
    results = Tuple{String,String,String,String,Vector{String},Bool}[]
    for (a, b, c, want) in questions
        missing_ = [w for w in (a, b, c, want) if !haskey(e.vocab, w)]
        if !isempty(missing_)
            push!(skipped, "$a/$b/$c/$want (missing " * join(missing_, ", ") * ")")
            continue
        end
        asked += 1
        got = first.(analogy(e, a, b, c; k = k))
        ok = want in got
        ok && (hits += 1)
        push!(results, (a, b, c, want, got, ok))
    end
    (accuracy = asked == 0 ? 0.0 : hits / asked, hits = hits, asked = asked,
     skipped = skipped, results = results)
end
