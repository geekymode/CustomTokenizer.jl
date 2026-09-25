"""
    AbstractEmbedding

Anything that maps words to vectors: a [`Model`](@ref) being trained here, or a
set of vectors loaded from a file with [`load_vectors`](@ref). The query
functions — [`wordvec`](@ref), [`similarity`](@ref),
[`nearest_neighbours`](@ref), [`analogy`](@ref) — work on either, so a trained
model and a published one can be compared with the same code.
"""
abstract type AbstractEmbedding end

"""
    Embedding(vocab, W; name="")

Word vectors without the training machinery: a vocabulary and one column per
word. Produced by [`load_vectors`](@ref), and accepted everywhere an
[`AbstractEmbedding`](@ref) is.
"""
struct Embedding <: AbstractEmbedding
    vocab::Vocabulary
    W::Matrix{Float64}
    name::String
end

Embedding(vocab::Vocabulary, W::AbstractMatrix; name::AbstractString = "") =
    Embedding(vocab, Matrix{Float64}(W), String(name))

Base.size(e::Embedding) = size(e.W)
Base.show(io::IO, e::Embedding) =
    print(io, "Embedding(", isempty(e.name) ? "" : e.name * ", ",
          size(e.W, 1), " × ", size(e.W, 2), ")")

"""
    embedding(model; name="") -> Embedding

The kept half of a [`Model`](@ref) on its own, so it can be saved or compared
without carrying `C` around.
"""
embedding(m::AbstractEmbedding; name::AbstractString = "") =
    Embedding(m.vocab, Matrix{Float64}(m.W), String(name))

"""
    load_vectors(path; max_words=0, name="") -> Embedding

Read word vectors in the usual plain-text format, one word per line:

    word  v1 v2 v3 ...

An optional first line giving `nwords dim` (the word2vec convention) is
detected and skipped; GloVe files, which omit it, load just as well. Lines
whose length disagrees with the first vector are skipped rather than trusted.

`max_words` reads only the first N entries. Published files are usually sorted
most-frequent-first, so that keeps the words that matter and skips the tail.

```julia
glove = load_vectors("glove-50d.vec"; max_words = 50_000, name = "GloVe 6B 50d")
nearest_neighbours(glove, "king", 5)
```
"""
function load_vectors(path::AbstractString; max_words::Integer = 0,
                      name::AbstractString = "")
    words = String[]
    cols = Vector{Float64}[]
    dim = 0
    open(path) do io
        first = true
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line)
            if first
                first = false
                # a header line is two integers and nothing else
                if length(parts) == 2 && all(p -> all(isdigit, p), parts)
                    continue
                end
            end
            length(parts) < 3 && continue
            dim == 0 && (dim = length(parts) - 1)
            length(parts) - 1 == dim || continue
            push!(words, String(parts[1]))
            push!(cols, parse.(Float64, parts[2:end]))
            max_words > 0 && length(words) >= max_words && break
        end
    end
    isempty(words) && throw(ArgumentError("no vectors found in $path"))
    W = reduce(hcat, cols)
    vocab = Vocabulary(words, Dict(w => i for (i, w) in enumerate(words)),
                       fill(0, length(words)))
    Embedding(vocab, W, isempty(name) ? basename(path) : String(name))
end

"""
    save_vectors(path, emb; header=true)

Write vectors in the same plain-text format, so a model trained here can be
loaded by other tools (gensim reads this directly) or reloaded later.
"""
function save_vectors(path::AbstractString, e::AbstractEmbedding; header::Bool = true)
    W, vocab = e.W, e.vocab
    open(path, "w") do io
        header && println(io, length(vocab), " ", size(W, 1))
        for (i, w) in enumerate(vocab.words)
            print(io, w)
            for d in 1:size(W, 1)
                @printf(io, " %.6f", W[d, i])
            end
            println(io)
        end
    end
    path
end
