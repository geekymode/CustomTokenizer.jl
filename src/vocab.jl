"""
    Vocabulary

Every distinct word, its integer id, and how often it occurs. Ids run from 1 and
are assigned most-frequent-first, ties broken alphabetically, so id 1 is usually
the commonest word in the corpus.

Fields: `words::Vector{String}`, `index::Dict{String,Int}`, `counts::Vector{Int}`.

Supports `length`, `v[i]` (id to word), `v["cat"]` (word to id), `haskey`,
and iteration over `(id, word, count)` triples.
"""
struct Vocabulary
    words::Vector{String}
    index::Dict{String,Int}
    counts::Vector{Int}
end

Base.length(v::Vocabulary) = length(v.words)
Base.getindex(v::Vocabulary, i::Integer) = v.words[i]
Base.getindex(v::Vocabulary, w::AbstractString) = v.index[w]
Base.haskey(v::Vocabulary, w::AbstractString) = haskey(v.index, w)
Base.iterate(v::Vocabulary, s::Int = 1) =
    s > length(v) ? nothing : ((s, v.words[s], v.counts[s]), s + 1)
Base.eltype(::Type{Vocabulary}) = Tuple{Int,String,Int}

function Base.show(io::IO, v::Vocabulary)
    print(io, "Vocabulary(", length(v), " words, ", sum(v.counts), " tokens)")
end

function Base.show(io::IO, ::MIME"text/plain", v::Vocabulary)
    println(io, "Vocabulary: ", length(v), " words, ", sum(v.counts), " tokens")
    n = min(length(v), 10)
    for i in 1:n
        @printf(io, "  %2d  %-10s %d\n", i, v.words[i], v.counts[i])
    end
    length(v) > n && print(io, "  ⋮  (", length(v) - n, " more)")
end

"""
    vocab_size(v) -> Int

Number of distinct words. Also the number of columns in both tables of a
[`Model`](@ref).
"""
vocab_size(v::Vocabulary) = length(v.words)

"""
    haskey_word(v, word) -> Bool

Whether `word` has an id. Convenience wrapper so callers need not reach into
`v.index`.
"""
haskey_word(v::Vocabulary, w::AbstractString) = haskey(v.index, w)

"""
    build_vocab(sentences; min_count=1) -> Vocabulary

Count every word, drop those occurring fewer than `min_count` times, and sort the
rest most-frequent-first (ties alphabetically).

`min_count` matters only on real corpora, where most word *types* occur once or
twice and carry no learnable signal; on the toy corpus the default keeps
everything.

```jldoctest
julia> v = build_vocab(tokenize(TOY_TEXT));

julia> length(v), v[1], v.counts[1]
(22, "the", 16)
```
"""
function build_vocab(sentences; min_count::Integer = 1)
    counts = Dict{String,Int}()
    for s in sentences, w in s
        counts[w] = get(counts, w, 0) + 1
    end
    kept = [(w, c) for (w, c) in counts if c >= min_count]
    sort!(kept; by = x -> (-x[2], x[1]))
    words = String[w for (w, _) in kept]
    Vocabulary(words, Dict(w => i for (i, w) in enumerate(words)), Int[c for (_, c) in kept])
end
