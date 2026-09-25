"""
    UpdateInfo

Everything one update did, returned by [`update_pair!`](@ref) so the step can be
inspected, printed or drawn.

Fields:
* `center`, `context`, `negatives` — the ids involved
* `words` — the same four words, as strings, context first
* `wants` — the correct answers: `1.0` for the context, `0.0` for the negatives
* `probabilities` — the guess `p` for each, *before* the update
* `errors` — `g = (want - p) * lr` for each, the amount each vector moves by
* `ΔW` — the change to the center's column of `W`
* `loss` — the cross-entropy of the four guesses
* `lr` — the learning rate used
"""
struct UpdateInfo
    center::Int
    context::Int
    negatives::Vector{Int}
    words::Vector{String}
    wants::Vector{Float64}
    probabilities::Vector{Float64}
    errors::Vector{Float64}
    ΔW::Vector{Float64}
    loss::Float64
    lr::Float64
end

function Base.show(io::IO, u::UpdateInfo)
    print(io, "UpdateInfo(center ", u.words[1], " ← ", u.words[2], ", loss ",
          round(u.loss; digits = 4), ")")
end

function Base.show(io::IO, ::MIME"text/plain", u::UpdateInfo)
    @printf(io, "update: center %s · context %s · negatives %s · lr %.4f\n",
            u.words[1], u.words[2], join(u.words[3:end], ", "), u.lr)
    println(io, "  word        want       p         g")
    for (k, w) in enumerate(u.words[2:end])
        @printf(io, "  %-10s %4.0f   %8.5f  %+8.5f\n", w, u.wants[k], u.probabilities[k], u.errors[k])
    end
    @printf(io, "  loss %.5f · |ΔW[:, %s]| %.3e\n", u.loss, u.words[1], norm(u.ΔW))
end

"""
    update_pair!(model, center, context, negatives, lr) -> UpdateInfo

One training step, the whole algorithm in six lines.

For the real context (`want = 1`) and each random word (`want = 0`):

1. guess `p = σ(W[:, center] ⋅ C[:, j])`
2. take the error `g = (want - p) * lr`
3. move the other word: `C[:, j] += g * W[:, center]`
4. collect the center's move: `grad += g * C[:, j]`, using `C` *before* step 3

and afterwards `W[:, center] += grad`.

Exactly one column of `W` and at most `1 + length(negatives)` columns of `C`
change; every other column is untouched. Each `C` change is a multiple of the
same vector `W[:, center]`, which makes the update rank one.

The returned [`UpdateInfo`](@ref) carries the guesses, the errors and the loss.
"""
function update_pair!(m::Model, center::Integer, context::Integer,
                 negatives::AbstractVector{<:Integer}, lr::Real)
    dim = size(m.W, 1)
    grad = zeros(dim)
    others = Int[context; negatives...]
    wants = Float64[k == 1 ? 1.0 : 0.0 for k in eachindex(others)]
    ps = zeros(length(others))
    gs = zeros(length(others))
    loss = 0.0
    w = view(m.W, :, center)
    for (k, o) in enumerate(others)
        c = view(m.C, :, o)
        p = sigmoid(dot(w, c))
        g = (wants[k] - p) * lr
        loss -= wants[k] == 1 ? log(max(p, eps())) : log(max(1 - p, eps()))
        @inbounds for d in 1:dim
            grad[d] += g * c[d]
            c[d] += g * w[d]
        end
        ps[k] = p
        gs[k] = g
    end
    @inbounds for d in 1:dim
        w[d] += grad[d]
    end
    words = String[m.vocab[center]; m.vocab[context]; [m.vocab[n] for n in negatives]]
    UpdateInfo(center, context, collect(negatives), words, wants, ps, gs, grad, loss, lr)
end

update_pair!(m::Model, center::AbstractString, context::AbstractString, negatives, lr::Real) =
    update_pair!(m, m.vocab[center], m.vocab[context],
            [n isa AbstractString ? m.vocab[n] : n for n in negatives], lr)

"""
    learning_rate(epoch, epochs; start=0.05, stop=0.001) -> Float64

The schedule used throughout: a linear decay from `start` to `stop` across
`epochs` passes. Early updates make coarse progress, late ones fine-tune, so the
vectors settle instead of jittering.
"""
learning_rate(epoch::Integer, epochs::Integer; start::Real = 0.05, stop::Real = 0.001) =
    start * (1 - (epoch - 1) / epochs) + stop

"""
    TrainLog

What [`train!`](@ref) recorded: `losses[e]` is the average loss per pair in epoch
`e`, and `snapshots[e]` holds copies of `(W, C)` for the epochs that were asked
for.
"""
struct TrainLog
    losses::Vector{Float64}
    snapshots::Dict{Int,NamedTuple{(:W, :C),Tuple{Matrix{Float64},Matrix{Float64}}}}
end

Base.show(io::IO, l::TrainLog) =
    print(io, "TrainLog(", length(l.losses), " epochs, final loss ",
          isempty(l.losses) ? "—" : round(l.losses[end]; digits = 3), ", ",
          length(l.snapshots), " snapshots)")

"""
    train!(model, pairs; epochs=300, sampler=UniformNegatives(), rng=Xoshiro(7),
           snapshots=Int[], lr_start=0.05, lr_stop=0.001, callback=nothing) -> TrainLog

Run [`update_pair!`](@ref) over every pair, `epochs` times, reshuffling the order each
epoch and drawing fresh negatives for every pair.

Shuffling matters: without it the vectors would pick up a rhythm from the
sentence order rather than from the counts.

`snapshots` lists epochs whose tables should be copied into the log (epoch `0`
means "before training"). `callback(epoch, model, loss)` is called after each
epoch, which is how the progress in the documentation is printed.
"""
function train!(m::Model, pairs::AbstractVector{Tuple{Int,Int}};
                epochs::Integer = 300, sampler::NegativeSampler = UniformNegatives(),
                rng::AbstractRNG = Xoshiro(7), snapshots::AbstractVector{<:Integer} = Int[],
                lr_start::Real = 0.05, lr_stop::Real = 0.001, callback = nothing)
    V = length(m.vocab)
    losses = Float64[]
    snaps = Dict{Int,NamedTuple{(:W, :C),Tuple{Matrix{Float64},Matrix{Float64}}}}()
    0 in snapshots && (snaps[0] = (W = copy(m.W), C = copy(m.C)))
    for epoch in 1:epochs
        lr = learning_rate(epoch, epochs; start = lr_start, stop = lr_stop)
        total = 0.0
        for (c, o) in shuffle(rng, pairs)
            negs = sample_negatives(sampler, rng, V, (c, o), m.negative)
            total += update_pair!(m, c, o, negs, lr).loss
        end
        push!(losses, total / length(pairs))
        epoch in snapshots && (snaps[epoch] = (W = copy(m.W), C = copy(m.C)))
        callback === nothing || callback(epoch, m, losses[end])
    end
    TrainLog(losses, snaps)
end

"""
    evaluate_loss(model, pairs; sampler=UniformNegatives(), rng=Xoshiro(1)) -> Float64

Average loss per pair *without* changing anything, for checking a model between
training runs.
"""
function evaluate_loss(m::Model, pairs::AbstractVector{Tuple{Int,Int}};
                       sampler::NegativeSampler = UniformNegatives(),
                       rng::AbstractRNG = Xoshiro(1))
    V = length(m.vocab)
    total = 0.0
    for (c, o) in pairs
        negs = sample_negatives(sampler, rng, V, (c, o), m.negative)
        p = probability(m, c, o)
        total -= log(max(p, eps()))
        for n in negs
            total -= log(max(1 - probability(m, c, n), eps()))
        end
    end
    total / length(pairs)
end
