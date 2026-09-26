"""
    CustomTokenizer

Skip-gram word2vec with negative sampling, written to be *read* rather than to be
fast: every step of the algorithm is a named function that can be called on its
own, inspected, plotted and tested.

The package grew out of a 12-sentence teaching script. That corpus is included as
[`TOY_TEXT`](@ref), and the defaults match it, but everything works on a real
corpus too — see the "Scaling up" page of the documentation.

A first run:

```julia
using CustomTokenizer

sentences = tokenize(TOY_TEXT)
vocab     = build_vocab(sentences)
model     = Model(vocab; dim = 8, window = 2, negative = 3)
pairs     = corpus_pairs(sentences, vocab; window = 2)

log = train!(model, pairs; epochs = 300)

similarity(model, "king", "queen")     # ≈ 1.00
nearest_neighbours(model, "cat", 3)    # dog, mouse, milk
```
"""
module CustomTokenizer

using LinearAlgebra, Printf, Random, Statistics

# corpus and vocabulary
export TOY_TEXT, tokenize, Vocabulary, build_vocab, vocab_size, haskey_word,
       subsample_probabilities, subsample

# pairs and counts
export context_pairs, corpus_pairs, cooccurrence, row_totals, pair_count

# embeddings: trained here, or loaded from a file
export AbstractEmbedding, Embedding, embedding, load_vectors, save_vectors

# comparing two embeddings
export shared_vocabulary, neighbour_overlap, similarity_agreement,
       analogy_accuracy, ANALOGY_QUESTIONS

# model
export Model, wordvec, ctxvec, score, probability, cosine_similarity, similarity,
       nearest_neighbours, analogy, column_norms

# negative sampling
export NegativeSampler, UniformNegatives, UnigramNegatives, sample_negatives,
       sampling_probabilities

# training
export UpdateInfo, update_pair!, learning_rate, train!, TrainLog, evaluate_loss

# stepping
export TrainStepper, step!, run_updates!, finish_epoch!, reset!, last_update, progress

# positional encoding
export sinusoidal_encoding, learned_positions, rope, alibi_slopes, alibi_bias,
       position_similarity, rope_similarity

# analysis
export target_probability, target_matrix, calibration, similarity_matrix, pca2,
       neighbour_ranking, ranking_churn

# plotting (implemented in the Makie extension)
export plot_tables, plot_update, plot_loss, plot_similarity_matrix,
       plot_cooccurrence, plot_embedding_map, plot_evolution,
       plot_positional_encoding, plot_position_decay

include("vocab.jl")
include("corpus.jl")
include("pairs.jl")
include("embedding.jl")
include("model.jl")
include("sampling.jl")
include("train.jl")
include("stepping.jl")
include("positional.jl")
include("analysis.jl")
include("compare.jl")
include("plots.jl")

end # module
