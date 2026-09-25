# Stepping through it

[`TrainStepper`](@ref) runs the same training one update at a time, so you can watch
the tables fill in instead of waiting for a result.

```@example step
using CustomTokenizer
sentences = tokenize(TOY_TEXT)
vocab     = build_vocab(sentences)
model     = Model(vocab)
pairs     = corpus_pairs(sentences, vocab; window = 2)

s = TrainStepper(model, pairs; seed = 7)
progress(s)
```

## The first update

```@example step
step!(s)
```

`W` did not move. Every `C` column this update touched was still zero, so the
center's push was a sum of zero vectors:

```@example step
last_update(s).ΔW
```

## When does `W` start moving?

Once some `C` column is non-zero, which happens almost immediately:

```@example step
first_move = 0
while first_move == 0
    info = step!(s)
    iszero(info.ΔW) || (global first_move = progress(s).update)
end
first_move
```

## How fast do the tables fill in?

```@example step
reset!(s)
nonzero_C() = count(j -> !iszero(@view model.C[:, j]), 1:length(vocab))
filled = Int[]
for _ in 1:20
    step!(s)
    push!(filled, nonzero_C())
end
filled'
```

Each update touches at most four `C` columns, so the table fills within the first
few updates of the first epoch.

## Bigger jumps

```@example step
run_updates!(s, 25)       # 25 updates, returns the last
progress(s)
```

```@example step
finish_epoch!(s)          # to the end of the epoch
progress(s)
```

## Replaying a run

[`reset!`](@ref) restores the starting tables and rewinds the counters, so the
same sequence can be replayed exactly:

```@example step
reset!(s)
run_updates!(s, 100)
snapshot = copy(model.W)

reset!(s)
run_updates!(s, 100)
model.W == snapshot
```

## Stepping and `train!` are the same rule

```@example step
a = Model(vocab); sa = TrainStepper(a, pairs; seed = 99); run_updates!(sa, length(pairs))
b = Model(vocab); train!(b, pairs; epochs = 1, rng = CustomTokenizer.Xoshiro(99))
a.W ≈ b.W && a.C ≈ b.C
```

## Watching a pair

Two probes tell the story: a pair that really occurs, and one that never does.

```@example step
reset!(s)
watch = [("cat", "drinks"), ("cat", "king")]
for _ in 1:20
    finish_epoch!(s)
end
[(a, b, round(probability(model, a, b); digits = 3)) for (a, b) in watch]
```

```@example step
for _ in 1:100
    finish_epoch!(s)
end
[(a, b, round(probability(model, a, b); digits = 3)) for (a, b) in watch]
```

The real pair heads for its balance point near 0.4, the impossible pair toward 0.
