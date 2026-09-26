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

## Embeddings: trained here or loaded from a file

```@docs
AbstractEmbedding
Embedding
embedding
load_vectors
save_vectors
```

## Comparing two embeddings

```@docs
shared_vocabulary
neighbour_overlap
similarity_agreement
analogy_accuracy
ANALOGY_QUESTIONS
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

## Real tokenizers

```@docs
BPETokenizer
load_hf_tokenizer
encode
decode
token_strings
token_id
token_string
```

(`vocab_size` works on a tokenizer too; it is documented under Vocabulary.)

## Checkpoints and downloads

```@docs
hf_download
safetensors_names
read_safetensor
embedding_from_checkpoint
```

## Positional encoding

```@docs
sinusoidal_encoding
learned_positions
rope
alibi_slopes
alibi_bias
position_similarity
rope_similarity
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
plot_positional_encoding
plot_position_decay
```
