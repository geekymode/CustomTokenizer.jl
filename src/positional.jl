# Positional encoding: how a model is told where a token sits.
#
# Attention has no notion of order. Scores are dot products between vectors, so
# permuting the input permutes the output and nothing else changes: "the cat
# chases the mouse" and "the mouse chases the cat" would be the same bag of
# vectors. Position has to be put in by hand, and the schemes below are the
# main ways of doing it.

"""
    sinusoidal_encoding(dim, len; base=10000.0) -> Matrix (dim × len)

The fixed encoding from *Attention is All You Need*: column `p` holds

```math
PE[2i,   p] = \\sin\\left(p / base^{2i/dim}\\right), \\qquad
PE[2i+1, p] = \\cos\\left(p / base^{2i/dim}\\right)
```

Each pair of rows is a clock, and the pairs run from fast (every couple of
positions) to slow (a whole context window). Reading a column is reading all
the clocks at once, which identifies the position the way hour and minute hands
identify a time.

Added to the token vectors before the first layer. It has no parameters, and
because the clocks are periodic, `PE[:, p+k]` is a fixed rotation of `PE[:, p]`
— the same rotation for every `p`, which is what lets attention pick up on
*relative* distance.

```jldoctest
julia> P = sinusoidal_encoding(4, 6);

julia> size(P)
(4, 6)

julia> round.(P[:, 1]; digits = 3)          # position 0: sin 0, cos 0
4-element Vector{Float64}:
 0.0
 1.0
 0.0
 1.0
```
"""
function sinusoidal_encoding(dim::Integer, len::Integer; base::Real = 10_000.0)
    iseven(dim) || throw(ArgumentError("dim must be even, got $dim"))
    P = zeros(Float64, dim, len)
    for p in 0:(len - 1), i in 0:(dim ÷ 2 - 1)
        θ = p / base^(2i / dim)
        P[2i + 1, p + 1] = sin(θ)
        P[2i + 2, p + 1] = cos(θ)
    end
    P
end

"""
    learned_positions(dim, maxlen; rng=Xoshiro(7), scale=0.02) -> Matrix (dim × maxlen)

The other classic choice: a plain lookup table, one column per position, learned
like any other parameter (GPT-2 and BERT do this).

Simple and effective, with one hard limit — position `maxlen` is the last column
that exists, so the model cannot be used beyond the length it was trained for.
The sinusoidal and rotary schemes have a value for every position, trained or
not.
"""
function learned_positions(dim::Integer, maxlen::Integer;
                           rng::AbstractRNG = Xoshiro(7), scale::Real = 0.02)
    randn(rng, dim, maxlen) .* scale
end

"""
    rope(x, positions; base=10000.0) -> Matrix

Rotary position embedding (RoPE), used by Llama, Gemma, Qwen and most current
models. Instead of adding anything to the token vector, it **rotates** it:
consecutive pairs of coordinates are treated as points in a plane and turned by
an angle proportional to the position.

```math
\\begin{pmatrix} x_{2i} \\\\ x_{2i+1} \\end{pmatrix} \\leftarrow
\\begin{pmatrix} \\cos p\\theta_i & -\\sin p\\theta_i \\\\
                 \\sin p\\theta_i & \\phantom{-}\\cos p\\theta_i \\end{pmatrix}
\\begin{pmatrix} x_{2i} \\\\ x_{2i+1} \\end{pmatrix},
\\qquad \\theta_i = base^{-2i/dim}
```

Two things follow, and both are worth checking rather than believing:

* rotation preserves length, so it cannot change how large a vector is;
* the dot product of a rotated query at position `m` with a rotated key at
  position `n` depends on `m - n` alone. Absolute positions cancel, leaving
  relative distance — exactly what attention wants.

`x` is `dim × n`, `positions` gives the position of each column (0-based).
Applied inside attention, to queries and keys, at every layer — not to the
token embeddings once.
"""
function rope(x::AbstractMatrix, positions::AbstractVector{<:Integer};
              base::Real = 10_000.0)
    dim, n = size(x)
    iseven(dim) || throw(ArgumentError("dim must be even, got $dim"))
    length(positions) == n ||
        throw(DimensionMismatch("got $(length(positions)) positions for $n columns"))
    y = similar(x, float(eltype(x)))
    for (col, p) in enumerate(positions), i in 0:(dim ÷ 2 - 1)
        θ = p / base^(2i / dim)
        c, s = cos(θ), sin(θ)
        a, b = x[2i + 1, col], x[2i + 2, col]
        y[2i + 1, col] = a * c - b * s
        y[2i + 2, col] = a * s + b * c
    end
    y
end

rope(x::AbstractVector, position::Integer; base::Real = 10_000.0) =
    vec(rope(reshape(x, :, 1), [position]; base = base))

"""
    alibi_slopes(nheads) -> Vector{Float64}

The per-head slopes of ALiBi (*Attention with Linear Biases*), geometrically
spaced so that different heads look at different distances: a steep head sees
only its neighbours, a shallow one sees the whole context.
"""
function alibi_slopes(nheads::Integer)
    nheads >= 1 || throw(ArgumentError("nheads must be positive"))
    closest = 2^floor(Int, log2(nheads))
    ratio = 2.0^(-8 / closest)
    slopes = [ratio^i for i in 1:closest]
    if closest < nheads
        extra_ratio = 2.0^(-4 / closest)
        append!(slopes, [extra_ratio^(2i - 1) for i in 1:(nheads - closest)])
    end
    slopes[1:nheads]
end

"""
    alibi_bias(nheads, len; causal=true) -> Array{Float64,3}  (len × len × nheads)

ALiBi adds nothing to the vectors at all. It puts a penalty straight into the
attention scores, proportional to how far apart the two positions are:

```math
\\text{score}_{ij} \\mathrel{+}= -m_h \\, |i - j|
```

With `causal = true` the upper triangle is `-Inf`, since a token cannot attend
to its future. The appeal is extrapolation: nothing was fitted to a particular
length, so a model trained on 2k tokens still behaves sensibly at 8k.
"""
function alibi_bias(nheads::Integer, len::Integer; causal::Bool = true)
    slopes = alibi_slopes(nheads)
    B = zeros(Float64, len, len, nheads)
    for h in 1:nheads, i in 1:len, j in 1:len
        B[i, j, h] = (causal && j > i) ? -Inf : -slopes[h] * abs(i - j)
    end
    B
end

"""
    position_similarity(P) -> Matrix

Cosine similarity between every pair of position vectors. For a sinusoidal
encoding the result is a band: nearby positions are similar, distant ones are
not, and the pattern depends mostly on the gap rather than on where you are —
which is the property that makes the encoding useful.
"""
function position_similarity(P::AbstractMatrix)
    n = size(P, 2)
    [cosine_similarity(view(P, :, i), view(P, :, j)) for i in 1:n, j in 1:n]
end

"""
    rope_similarity(dim, len; base=10000.0, samples=64, rng=Xoshiro(7)) -> Vector

How a rotated dot product decays with distance. For each gap `0:len-1` it takes
random unit vectors `v` and averages `dot(rope(v, 0), rope(v, gap))`: the score
a token would give to an identical token that many positions away.

The same vector is used on both sides deliberately. Two *independent* random
vectors have a dot product near zero at every gap, so nothing would show;
comparing a vector with itself isolates what position alone does. At gap 0 the
value is 1, and it falls away as the faster clocks drift out of phase.
"""
function rope_similarity(dim::Integer, len::Integer; base::Real = 10_000.0,
                         samples::Integer = 64, rng::AbstractRNG = Xoshiro(7))
    vs = [normalize(randn(rng, dim)) for _ in 1:samples]
    [mean(dot(rope(v, 0; base = base), rope(v, gap; base = base)) for v in vs)
     for gap in 0:(len - 1)]
end
