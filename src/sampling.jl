"""
    NegativeSampler

How the "not a neighbour" examples are drawn. Two implementations are provided:
[`UniformNegatives`](@ref), which every word is equally likely to win, and
[`UnigramNegatives`](@ref), which follows word frequency raised to a power, as
real word2vec does.

The choice matters more than it looks: with a large vocabulary a uniform draw
almost always lands on a word too rare to teach anything.
"""
abstract type NegativeSampler end

"""
    UniformNegatives()

Draw uniformly from the vocabulary, excluding the center and context of the
current pair. Suitable for a toy vocabulary; `O(V)` per draw, since it builds the
complement explicitly, exactly as the teaching script does.
"""
struct UniformNegatives <: NegativeSampler end

"""
    UnigramNegatives(vocab; power=0.75, table_size=1_000_000)

Draw in proportion to `count^power`, the rule used by word2vec. Frequent words
are drawn often enough to be informative, rare ones still appear. Implemented
with a precomputed lookup table, so a draw is `O(1)`.
"""
struct UnigramNegatives <: NegativeSampler
    table::Vector{Int}
    probabilities::Vector{Float64}
end

function UnigramNegatives(vocab::Vocabulary; power::Real = 0.75,
                          table_size::Integer = 1_000_000)
    p = Float64.(vocab.counts) .^ power
    p ./= sum(p)
    cdf = cumsum(p)
    V = length(vocab)
    table = [min(V, searchsortedfirst(cdf, (i - 0.5) / table_size)) for i in 1:table_size]
    UnigramNegatives(table, p)
end

"""
    sampling_probabilities(sampler, V) -> Vector{Float64}

Probability of each word being picked for one negative draw, as
[`target_probability`](@ref) needs it.

Both samplers exclude the two words of the current pair, so for
[`UniformNegatives`](@ref) an eligible word has probability `1/(V-2)`, not `1/V`.
For [`UnigramNegatives`](@ref) the `count^power` probabilities are returned as
they are; the correction for the two excluded ids is negligible on any
vocabulary large enough to want that sampler.
"""
sampling_probabilities(::UniformNegatives, V::Integer) = fill(1 / (V - 2), V)
sampling_probabilities(s::UnigramNegatives, ::Integer) = copy(s.probabilities)

"""
    sample_negatives(sampler, rng, V, avoid, k) -> Vector{Int}

Draw `k` negative word ids. Draws are independent, so the same word can come up
twice in one update; word2vec does not check for that, and neither does this.

`avoid` lists ids that must not be drawn (the center and the real context).
"""
function sample_negatives(::UniformNegatives, rng::AbstractRNG, V::Integer,
                          avoid, k::Integer)
    pool = setdiff(1:V, avoid)
    [rand(rng, pool) for _ in 1:k]
end

function sample_negatives(s::UnigramNegatives, rng::AbstractRNG, V::Integer,
                          avoid, k::Integer)
    out = Int[]
    while length(out) < k
        w = s.table[rand(rng, 1:length(s.table))]
        w in avoid && continue
        push!(out, w)
    end
    out
end
