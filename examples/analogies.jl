# "man is to king as woman is to ?" — vector arithmetic on trained vectors.
#
# Analogies need a real corpus: a few thousand words cannot support them. This
# trains on a slice of text8 (Wikipedia) and then asks a set of questions.
#
#   julia --project=.. examples/analogies.jl ../text8 2000000
#
# With no arguments it falls back to any corpus file given, or to the Lee news
# corpus, which is far too small — that failure is itself the lesson.

using CustomTokenizer, Printf, Random, LinearAlgebra

corpus = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "text8")
ntokens = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2_000_000
isfile(corpus) || error("corpus not found: $corpus")

const DIM, WINDOW, NEG, EPOCHS, MIN_COUNT = 100, 5, 5, 3, 5

# ---------------------------------------------------------------- corpus
raw = open(corpus) do io
    String(read(io, min(filesize(corpus), 7 * ntokens)))
end
words = [String(m.match) for m in eachmatch(r"[A-Za-z]+", lowercase(raw))]
words = words[1:min(ntokens, length(words))]
sentences = [words[i:min(i + 999, length(words))] for i in 1:1000:length(words)]
@printf("corpus: %d tokens in %d blocks\n", length(words), length(sentences))

vocab = build_vocab(sentences; min_count = MIN_COUNT)
@printf("vocabulary: %d words (min_count %d)\n", length(vocab), MIN_COUNT)

# Drop most occurrences of the very frequent words, as real word2vec does:
# without this, "the" dominates the pairs and the vectors stay blurred.
keep = subsample_probabilities(vocab; threshold = 1e-4)
ids = subsample(sentences, vocab, keep, Xoshiro(7))
pairs = corpus_pairs(ids; window = WINDOW)
@printf("after subsampling: %d tokens, %d pairs\n", sum(length, ids), length(pairs))

# ---------------------------------------------------------------- train
model = Model(vocab; dim = DIM, window = WINDOW, negative = NEG, rng = Xoshiro(7))
sampler = UnigramNegatives(vocab; table_size = 1_000_000)
@printf("training %d epochs ...\n", EPOCHS)
t = @elapsed trainlog = train!(model, pairs; epochs = EPOCHS, sampler = sampler, rng = Xoshiro(7))
@printf("  %.1f s, loss %.3f -> %.3f\n\n", t, trainlog.losses[1], trainlog.losses[end])

# ---------------------------------------------------------------- ask
"""
    ask(a, b, c; k)

`a` is to `b` as `c` is to ? — the words closest to `b - a + c`, with the three
inputs excluded (they are always closest to their own combination).
"""
function ask(a, b, c; k = 4)
    for w in (a, b, c)
        haskey(model.vocab, w) || return "not in the vocabulary: $w"
    end
    join((@sprintf("%s (%.2f)", w, s) for (w, s) in analogy(model, a, b, c; k = k)), ", ")
end

questions = [
    ("man", "king", "woman", "queen"),
    ("woman", "queen", "man", "king"),
    ("woman", "aunt", "man", "uncle"),
    ("man", "uncle", "woman", "aunt"),
    ("boy", "son", "girl", "daughter"),
    ("france", "paris", "italy", "rome"),
    ("germany", "berlin", "japan", "tokyo"),
    ("good", "better", "bad", "worse"),
    ("walk", "walking", "swim", "swimming"),
]

println("=== a is to b as c is to ? ===\n")
hits = 0
for (a, b, c, want) in questions
    got = ask(a, b, c)
    first_word = split(got, " (")[1]
    mark = first_word == want ? "✓" : occursin(want * " (", got) ? "~" : " "
    first_word == want && (global hits += 1)
    @printf("%s %-7s → %-8s as %-8s → ?   want %-9s got %s\n", mark, a, b, c, want, got)
end
@printf("\n%d of %d correct at rank 1 (~ means the answer is in the top %d)\n",
        hits, length(questions), 4)

println("\n=== the same words, as plain neighbours ===\n")
for w in ("king", "aunt", "paris")
    haskey(model.vocab, w) || continue
    @printf("%-7s %s\n", w,
            join((@sprintf("%s (%.2f)", x, s) for (x, s) in nearest_neighbours(model, w, 5)), ", "))
end

println("""

Analogies are a harder test than neighbours: they need the *difference*
between two words to be a consistent direction across the whole vocabulary.
A few million tokens gives a partial result; the published word2vec numbers
come from billions.""")
