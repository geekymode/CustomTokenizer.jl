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

# ---------------------------------------------------------------- the clocks
"""
    sinusoidal_frequencies(dim; base=10000.0) -> Vector

The angular frequencies ``\\theta_i = base^{-2i/dim}`` behind both
[`sinusoidal_encoding`](@ref) and [`rope`](@ref) — one per coordinate *pair*,
so `dim ÷ 2` of them. They form a geometric progression from 1 down to
`1/base`, which is what makes the scheme scale-free: each pair covers its own
octave of distance.
"""
function sinusoidal_frequencies(dim::Integer; base::Real = 10_000.0)
    iseven(dim) || throw(ArgumentError("dim must be even, got $dim"))
    [base^(-2i / dim) for i in 0:(dim ÷ 2 - 1)]
end

"""
    sinusoidal_wavelengths(dim; base=10000.0) -> Vector

How many positions each pair takes to come back round: ``\\lambda_i =
2\\pi/\\theta_i``. At `dim = 64, base = 10000` these run from 6.3 positions to
about 47,000 — the fast pairs resolve neighbours, the slow ones place a token
in the document.

A pair is only informative while `gap < λ/2`; past that it has wrapped and its
reading is ambiguous, exactly like the hour hand of a clock at 13:00.
"""
sinusoidal_wavelengths(dim::Integer; base::Real = 10_000.0) =
    2π ./ sinusoidal_frequencies(dim; base = base)

"""
    sinusoidal_gap_score(dim, gaps; base=10000.0) -> Vector

The exact similarity kernel of the sinusoidal encoding:

```math
S(g) \\;=\\; PE[:,p] \\cdot PE[:,p+g] \\;=\\; \\sum_i \\cos(g\\,\\theta_i)
```

The absolute position cancels — the two columns' dot product depends on the
gap alone, so the similarity matrix is Toeplitz (constant along diagonals).

Worth knowing before trusting the usual picture: `S` is **not** a clean
monotone decay. It is a sum of `dim ÷ 2` cosines at geometrically spaced
frequencies, so past the first few positions it oscillates. At `dim = 64` it
decreases only out to a gap of about 6, and beyond gap 500 it wanders between
roughly −6 and +13 against a peak of `S(0) = 32`. The smooth curve everyone
draws is a large-dimension limit; see [`sinusoidal_gap_envelope`](@ref).

```jldoctest
julia> round.(sinusoidal_gap_score(64, [0, 1, 2]); digits = 3)
3-element Vector{Float64}:
 32.0
 30.917
 28.304
```
"""
function sinusoidal_gap_score(dim::Integer, gaps; base::Real = 10_000.0)
    θ = sinusoidal_frequencies(dim; base = base)
    [sum(cos(g * t) for t in θ) for g in gaps]
end

"""
    sinusoidal_gap_envelope(dim, gaps; base=10000.0) -> Vector

The smooth approximation to [`sinusoidal_gap_score`](@ref), obtained by
treating the sum over pairs as an integral:

```math
S(g) \\;\\approx\\; \\frac{dim}{2}\\left(1 - \\frac{\\gamma + \\ln g}{\\ln base}\\right)
```

so the similarity is predicted to fall **logarithmically** in the gap — equal
loss per octave of distance, which is the sense in which the encoding is
scale-free.

It is an asymptotic in the dimension, and a mediocre one at the sizes actually
used. Mean relative error over gaps 10–1000: 0.105 at `dim = 32`, 0.066 at 64,
0.024 at 256, 0.007 at 1024, and 0.000 by 4096. Use it as the envelope of the
true kernel, never as a substitute for it.
"""
function sinusoidal_gap_envelope(dim::Integer, gaps; base::Real = 10_000.0)
    γ = 0.5772156649015329          # Euler–Mascheroni
    [g <= 0 ? dim / 2 : (dim / 2) * (1 - (γ + log(g)) / log(base)) for g in gaps]
end

"""
    shift_operator(dim, k; base=10000.0) -> Matrix (dim × dim)

The matrix `R` with `R * PE[:, p] == PE[:, p + k]` for **every** `p`, where
`PE` is a [`sinusoidal_encoding`](@ref). Translating a position is a fixed
linear map — that is the whole reason the encoding is useful, and it follows
from the angle-addition formulas:

```math
\\sin((p+k)\\theta) = \\sin p\\theta \\cos k\\theta + \\cos p\\theta \\sin k\\theta, \\qquad
\\cos((p+k)\\theta) = \\cos p\\theta \\cos k\\theta - \\sin p\\theta \\sin k\\theta
```

`R` is block-diagonal with one 2×2 block per pair and is orthogonal, so the
shift is a rotation: it moves a column around the torus without changing its
length.

Rows here are ordered `(sin, cos)`, which puts the block in the form
`[cos kθ  sin kθ; −sin kθ  cos kθ]`.
"""
function shift_operator(dim::Integer, k::Integer; base::Real = 10_000.0)
    θ = sinusoidal_frequencies(dim; base = base)
    R = zeros(Float64, dim, dim)
    for (j, t) in enumerate(θ)
        i = j - 1
        c, s = cos(k * t), sin(k * t)
        R[2i + 1, 2i + 1] =  c;  R[2i + 1, 2i + 2] = s
        R[2i + 2, 2i + 1] = -s;  R[2i + 2, 2i + 2] = c
    end
    R
end

"""
    rotation_operator(dim, p; base=10000.0) -> Matrix (dim × dim)

RoPE's rotation `R(p)` written out as a matrix, so the algebra can be checked
rather than taken on trust. `rotation_operator(dim, p) * x == rope(x, p)`.

`R` is a *representation of the integers under addition*: `R(a)R(b) = R(a+b)`
and `R(a)' == R(-a)`, every `R(p)` orthogonal. Those two lines are the entire
justification for RoPE, because they give

```math
\\langle R(m)q,\\; R(n)k \\rangle = q^{\\!\\top} R(m)^{\\!\\top} R(n) k
  = q^{\\!\\top} R(n-m) k
```

— the score depends on `n - m` and nothing else.
"""
function rotation_operator(dim::Integer, p::Integer; base::Real = 10_000.0)
    θ = sinusoidal_frequencies(dim; base = base)
    R = zeros(Float64, dim, dim)
    for (j, t) in enumerate(θ)
        i = j - 1
        c, s = cos(p * t), sin(p * t)
        R[2i + 1, 2i + 1] = c;  R[2i + 1, 2i + 2] = -s
        R[2i + 2, 2i + 1] = s;  R[2i + 2, 2i + 2] =  c
    end
    R
end

"""
    rope_channels(q, k; base=10000.0) -> (amplitude, phase, frequency)

Decomposes a RoPE attention score into one cosine per coordinate pair. Reading
pair `i` of `q` as a complex number `zᵢ` and of `k` as `wᵢ`,

```math
\\langle R(m)q,\\; R(n)k \\rangle
  = \\sum_i |z_i|\\,|w_i| \\, \\cos\\!\\big(g\\,\\theta_i + \\varphi_i\\big),
\\qquad g = n-m,\\;\\; \\varphi_i = \\arg w_i - \\arg z_i
```

This is the clearest statement of what RoPE does. **Content chooses each
channel's amplitude and phase; position only advances the argument.** The score
is a filter bank over relative distance, and a head can tune itself to respond
at a chosen gap by choosing the phases — something no additive scheme offers.

Returns the three vectors, one entry per pair.
"""
function rope_channels(q::AbstractVector, k::AbstractVector; base::Real = 10_000.0)
    length(q) == length(k) ||
        throw(DimensionMismatch("q has $(length(q)) entries, k has $(length(k))"))
    dim = length(q)
    θ = sinusoidal_frequencies(dim; base = base)
    amp   = similar(θ)
    phase = similar(θ)
    for (j, _) in enumerate(θ)
        i = j - 1
        z = complex(q[2i + 1], q[2i + 2])
        w = complex(k[2i + 1], k[2i + 2])
        amp[j]   = abs(z) * abs(w)
        phase[j] = angle(w) - angle(z)
    end
    (amplitude = amp, phase = phase, frequency = θ)
end

"""
    alibi_halflife(nheads) -> Vector

How many positions it takes each ALiBi head to halve its attention:
`ln 2 / mₕ`.

The bias `-mₕ|i-j|` sits inside a softmax, so after exponentiating it is a
**geometric discount**: attention is multiplied by `r^{|i-j|}` with
`r = exp(-mₕ)`. ALiBi is therefore a recency prior with a per-head half-life,
and because the slopes are geometric the half-lives are too — at 8 heads they
run 1.4, 2.8, 5.5, 11.1, 22.2, 44.4, 88.7, 177.4 positions. The same
octave-tiling idea as the sinusoidal clocks, applied to the attention edges
instead of the vectors.
"""
alibi_halflife(nheads::Integer) = log(2) ./ alibi_slopes(nheads)

"""
    alibi_decay(nheads, len) -> Matrix (len × nheads)

The multiplicative weight ALiBi applies at each distance, `exp(-mₕ d)` for
`d = 0:len-1` — the same information as [`alibi_bias`](@ref), on the scale the
softmax actually sees. Column `h` is one head's decay curve, falling from 1 to
`exp(-mₕ(len-1))`.
"""
function alibi_decay(nheads::Integer, len::Integer)
    slopes = alibi_slopes(nheads)
    [exp(-slopes[h] * d) for d in 0:(len - 1), h in 1:nheads]
end
