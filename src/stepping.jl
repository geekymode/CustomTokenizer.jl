"""
    TrainStepper(model, pairs; sampler=UniformNegatives(), rng=Xoshiro(7), epochs=300,
            lr_start=0.05, lr_stop=0.001)

Training, one update at a time, so a run can be watched instead of waited for.

The stepper owns the shuffled order of the current epoch and the position within
it, and applies exactly the same rule as [`train!`](@ref):

```julia
s = TrainStepper(model, pairs)
step!(s)             # one update, returns an UpdateInfo
run_updates!(s, 25)  # 25 updates, returns the last one
finish_epoch!(s)     # to the end of the current epoch
reset!(s)            # back to the start, tables restored
progress(s)         # (update = 354, epoch = 3, pair = 1, of = 176)
```

`reset!` needs the model's starting tables, so the stepper copies them when it is
built.
"""
mutable struct TrainStepper
    model::Model
    pairs::Vector{Tuple{Int,Int}}
    sampler::NegativeSampler
    rng::AbstractRNG
    epochs::Int
    lr_start::Float64
    lr_stop::Float64
    order::Vector{Tuple{Int,Int}}
    epoch::Int
    k::Int
    n::Int
    seed::UInt64
    W0::Matrix{Float64}
    C0::Matrix{Float64}
    last::Union{Nothing,UpdateInfo}
end

function TrainStepper(m::Model, pairs::AbstractVector{Tuple{Int,Int}};
                 sampler::NegativeSampler = UniformNegatives(),
                 seed::Integer = 7, epochs::Integer = 300,
                 lr_start::Real = 0.05, lr_stop::Real = 0.001)
    TrainStepper(m, collect(pairs), sampler, Xoshiro(seed), epochs, lr_start, lr_stop,
            Tuple{Int,Int}[], 0, 0, 0, UInt64(seed), copy(m.W), copy(m.C), nothing)
end

function Base.show(io::IO, s::TrainStepper)
    print(io, "TrainStepper(update ", s.n, ", epoch ", s.epoch, ", pair ", s.k, " of ",
          length(s.pairs), ")")
end

"""
    progress(stepper) -> NamedTuple

Where the run is: `(update, epoch, pair, of)`.
"""
progress(s::TrainStepper) = (update = s.n, epoch = s.epoch, pair = s.k, of = length(s.pairs))

"""
    last_update(stepper) -> Union{Nothing,UpdateInfo}

The most recent [`UpdateInfo`](@ref), or `nothing` before the first step.
"""
last_update(s::TrainStepper) = s.last

"""
    step!(stepper) -> UpdateInfo

Apply the next pair. Starts a new epoch (reshuffling the pairs and lowering the
learning rate) when the current one runs out.
"""
function step!(s::TrainStepper)
    if s.k == length(s.order)
        s.epoch += 1
        s.order = shuffle(s.rng, s.pairs)
        s.k = 0
    end
    s.k += 1
    c, o = s.order[s.k]
    lr = learning_rate(s.epoch, s.epochs; start = s.lr_start, stop = s.lr_stop)
    negs = sample_negatives(s.sampler, s.rng, length(s.model.vocab), (c, o), s.model.negative)
    info = update_pair!(s.model, c, o, negs, lr)
    s.n += 1
    s.last = info
    info
end

"""
    run_updates!(stepper, n=1) -> UpdateInfo

Apply `n` updates and return the last one.
"""
function run_updates!(s::TrainStepper, n::Integer = 1)
    n >= 1 || throw(ArgumentError("n must be at least 1"))
    info = step!(s)
    for _ in 2:n
        info = step!(s)
    end
    info
end

"""
    finish_epoch!(stepper) -> UpdateInfo

Run to the end of the current epoch, or through a whole new one if the current
epoch has just finished.
"""
function finish_epoch!(s::TrainStepper)
    remaining = s.k == length(s.order) ? length(s.pairs) : length(s.order) - s.k
    run_updates!(s, remaining)
end

"""
    reset!(stepper)

Put the tables back to their starting values and rewind the counters, so the same
sequence of updates can be replayed.
"""
function reset!(s::TrainStepper)
    s.model.W .= s.W0
    s.model.C .= s.C0
    s.rng = Xoshiro(s.seed)
    s.order = Tuple{Int,Int}[]
    s.epoch = 0
    s.k = 0
    s.n = 0
    s.last = nothing
    s
end

"""
    changed_columns(info) -> NamedTuple

Which columns an update touched: `(W = [center], C = unique(context and
negatives))`. Useful in tests and in the plots that outline them.
"""
changed_columns(u::UpdateInfo) = (W = [u.center], C = unique(Int[u.context; u.negatives]))
