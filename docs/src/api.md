# API reference

```@meta
CurrentModule = CustomTokenizer
```

## Module

```@docs
CustomTokenizer
```

## Corpus

```@docs
TOY_TEXT
tokenize
subsample_probabilities
subsample
```

## Vocabulary

```@docs
Vocabulary
build_vocab
vocab_size
haskey_word
```

## Pairs and counts

```@docs
context_pairs
corpus_pairs
cooccurrence
row_totals
pair_count
```

## The model

```@docs
Model
wordvec
ctxvec
score
probability
cosine_similarity
similarity
nearest_neighbours
analogy
column_norms
```

## Negative sampling

```@docs
NegativeSampler
UniformNegatives
UnigramNegatives
sample_negatives
sampling_probabilities
```

## Training

```@docs
UpdateInfo
update_pair!
learning_rate
TrainLog
train!
evaluate_loss
```

## Stepping

```@docs
TrainStepper
step!
run_updates!
finish_epoch!
reset!
progress
last_update
```

## Analysis

```@docs
target_probability
target_matrix
calibration
similarity_matrix
pca2
neighbour_ranking
ranking_churn
```

## Plotting

These methods need a Makie backend (`using CairoMakie`).

```@docs
plot_tables
plot_update
plot_loss
plot_similarity_matrix
plot_cooccurrence
plot_embedding_map
plot_evolution
```
