"""
    target_probability(N, totals, probs, i, j, negative) -> Float64

Where training will settle for the pair `(i, j)`, from the counts alone.

Per epoch the pair receives `N[i,j]` *pulls*, one per occurrence, and on average
`negative * (totals[i] - N[i,j]) * probs[j]` *pushes*, from the times `j` is
drawn as a random word while `i` is the center. The two cancel at

    p* = pulls / (pulls + pushes)

so the model's probability for a pair is not "how often do these occur" but "how
much more often than chance".
"""
function target_probability(N::AbstractMatrix{<:Integer}, totals::AbstractVector{<:Integer},
                            probs::AbstractVector{<:Real}, i::Integer, j::Integer,
                            negative::Integer)
    n = N[i, j]
    n == 0 && return 0.0
    pushes = negative * (totals[i] - n) * probs[j]
    n / (n + pushes)
end

"""
    target_matrix(N, probs, negative) -> Matrix{Float64}

[`target_probability`](@ref) for every pair. The diagonal is set to `NaN`, since
a word is never its own neighbour.
"""
function target_matrix(N::AbstractMatrix{<:Integer}, probs::AbstractVector{<:Real},
                       negative::Integer)
    totals = row_totals(N)
    V = size(N, 1)
    T = [i == j ? NaN : target_probability(N, totals, probs, i, j, negative) for i in 1:V, j in 1:V]
    T
end

"""
    calibration(model, N, probs) -> NamedTuple

How closely the trained model reproduces those targets:
`(correlation, mean_absolute_error, n)` over all off-diagonal pairs.

A well-trained toy run reaches a correlation above 0.99, which is the sense in
which word2vec "is" a factorisation of the co-occurrence table.
"""
function calibration(m::Model, N::AbstractMatrix{<:Integer}, probs::AbstractVector{<:Real})
    V = length(m.vocab)
    totals = row_totals(N)
    x = Float64[]
    y = Float64[]
    for i in 1:V, j in 1:V
        i == j && continue
        push!(x, target_probability(N, totals, probs, i, j, m.negative))
        push!(y, probability(m, i, j))
    end
    (correlation = cor(x, y), mean_absolute_error = mean(abs.(x .- y)), n = length(x))
end

"""
    similarity_matrix(model; order=1:length(model.vocab)) -> Matrix{Float64}

Cosine similarity between every pair of word vectors, optionally reordered (pass
a permutation to group related words together and make the block structure
visible).
"""
function similarity_matrix(m::Model; order = 1:length(m.vocab))
    idx = collect(order)
    [similarity(m, i, j) for i in idx, j in idx]
end

"""
    pca2(M; basis=nothing, centre=nothing) -> (coords, basis, centre)

Flatten columns to two dimensions for drawing. Each column is scaled to length 1
first, so only direction matters — the quantity cosine similarity measures.

Passing the `basis` and `centre` returned by an earlier call projects a different
snapshot onto the *same* axes, which is what makes a sequence of frames
comparable.
"""
function pca2(M::AbstractMatrix; basis = nothing, centre = nothing)
    U = M ./ sqrt.(sum(abs2, M, dims = 1))
    c = centre === nothing ? sum(U, dims = 2) ./ size(U, 2) : centre
    X = U .- c
    B = basis === nothing ? svd(X).U[:, 1:2] : basis
    (B' * X, B, c)
end

"""
    neighbour_ranking(model, k=3) -> Vector{Set{Int}}

For each word, the ids of its `k` nearest neighbours. Comparing two rankings with
[`ranking_churn`](@ref) says whether training has stopped rearranging things.
"""
function neighbour_ranking(m::Model, k::Integer = 3)
    V = length(m.vocab)
    [Set(partialsortperm([j == i ? -Inf : similarity(m, i, j) for j in 1:V], 1:k; rev = true))
     for i in 1:V]
end

"""
    ranking_churn(a, b) -> Int

How many entries moved in or out across two [`neighbour_ranking`](@ref) results:
zero means the neighbour lists are identical.
"""
ranking_churn(a::AbstractVector{<:AbstractSet}, b::AbstractVector{<:AbstractSet}) =
    sum(length(setdiff(b[i], a[i])) for i in eachindex(a))
