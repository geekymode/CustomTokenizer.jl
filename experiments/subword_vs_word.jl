# Does the tokenizer change the vectors?
#
# Trains the same model twice on the same corpus: once on whole words (this
# package's tokenizer) and once on Gemma 4 sub-word pieces. A word vector in the
# sub-word run is the mean of the vectors of its pieces.
#
#   julia --project=. experiments/tokenize_corpus.jl ../lee_background.cor
#   julia --project=docs experiments/subword_vs_word.jl

using CustomTokenizer, CairoMakie, Printf, Random, Statistics, LinearAlgebra

CairoMakie.activate!(type = "png")
here    = @__DIR__
results = joinpath(here, "results")
figs    = joinpath(here, "figures"); mkpath(figs)

# Words below MIN_COUNT are dropped from the whole-word vocabulary, as usual.
# Pieces are NOT filtered: a real tokenizer's vocabulary is fixed in advance,
# so every piece that occurs is one the model already has an entry for. That
# asymmetry is the point of the comparison.
const MIN_COUNT        = 5
const MIN_COUNT_PIECES = 1
const DIM       = 64
const WINDOW    = 5
const NEGATIVE  = 5
const EPOCHS    = 5

read_stream(path) = [split(line) |> collect for line in eachline(path) if !isempty(strip(line))]

"word => the pieces it is built from"
function read_pieces(path)
    d = Dict{String,Vector{String}}()
    for line in eachline(path)
        parts = split(line, '\t')
        length(parts) == 2 && (d[parts[1]] = String.(split(parts[2])))
    end
    d
end

function train_on(sentences; seed = 7, min_count = MIN_COUNT)
    vocab = build_vocab(sentences; min_count = min_count)
    pairs = corpus_pairs(sentences, vocab; window = WINDOW)
    model = Model(vocab; dim = DIM, window = WINDOW, negative = NEGATIVE, rng = Xoshiro(seed))
    sampler = UnigramNegatives(vocab; table_size = 200_000)
    log = train!(model, pairs; epochs = EPOCHS, sampler = sampler, rng = Xoshiro(seed))
    (model = model, vocab = vocab, pairs = pairs, log = log)
end

"A word vector in the sub-word run: the mean of its pieces' vectors."
function pooled(model, word, table)
    ps = get(table, word, String[])
    vs = [wordvec(model, p) for p in ps if haskey(model.vocab, p)]
    isempty(vs) ? nothing : normalize(sum(vs) ./ length(vs))
end

function neighbours_of(word, words, vecs; k = 5)
    v = get(vecs, word, nothing)
    v === nothing && return Tuple{String,Float64}[]
    sims = [(w, dot(v, vecs[w])) for w in words if w != word && haskey(vecs, w)]
    sort!(sims; by = x -> -x[2])
    sims[1:min(k, length(sims))]
end

# ---------------------------------------------------------------- run both
println("reading the two token streams ...")
word_sents  = read_stream(joinpath(results, "stream_word.txt"))
piece_sents = read_stream(joinpath(results, "stream_gemma.txt"))
pieces_of   = read_pieces(joinpath(results, "word_pieces.tsv"))

counts = Dict{String,Int}()
for s in word_sents, w in s; counts[w] = get(counts, w, 0) + 1; end
all_words = collect(keys(counts))

@printf("word stream : %d sentences, %d tokens\n", length(word_sents), sum(length, word_sents))
@printf("piece stream: %d sentences, %d tokens\n", length(piece_sents), sum(length, piece_sents))

println("\ntraining on whole words ...")
t1 = @elapsed W = train_on(word_sents)
@printf("  %d words kept, %d pairs, %.1f s, loss %.3f -> %.3f\n",
        length(W.vocab), length(W.pairs), t1, W.log.losses[1], W.log.losses[end])

println("training on Gemma pieces ...")
t2 = @elapsed S = train_on(piece_sents; min_count = MIN_COUNT_PIECES)
@printf("  %d pieces kept, %d pairs, %.1f s, loss %.3f -> %.3f\n",
        length(S.vocab), length(S.pairs), t2, S.log.losses[1], S.log.losses[end])

# ---------------------------------------------------------------- coverage
println("\n=== which words can each scheme represent? ===\n")
rare = [w for w in all_words if counts[w] < MIN_COUNT]
cov_word = count(w -> haskey(W.vocab, w), all_words) / length(all_words)
cov_sub  = count(w -> pooled(S.model, w, pieces_of) !== nothing, all_words) / length(all_words)
rare_sub = count(w -> pooled(S.model, w, pieces_of) !== nothing, rare) / max(1, length(rare))
@printf("  corpus words                      %d\n", length(all_words))
@printf("  seen fewer than %d times           %d (%.0f%%)\n", MIN_COUNT, length(rare),
        100length(rare) / length(all_words))
@printf("  representable, whole-word model   %.1f%%\n", 100cov_word)
@printf("  representable, sub-word model     %.1f%%\n", 100cov_sub)
@printf("  of the rare words, sub-word covers %.1f%%\n", 100rare_sub)

# ---------------------------------------------------------------- morphology
println("\n=== do related forms end up closer? ===\n")
suffixes = ("s", "es", "ed", "ing")
families = Tuple{String,String}[]
for w in all_words, suf in suffixes
    infl = w * suf
    haskey(counts, infl) && counts[w] >= MIN_COUNT && counts[infl] >= MIN_COUNT &&
        push!(families, (w, infl))
end
sort!(families)
vec_word = Dict(w => normalize(collect(wordvec(W.model, w))) for w in all_words if haskey(W.vocab, w))
vec_sub  = Dict{String,Vector{Float64}}()
for w in all_words
    v = pooled(S.model, w, pieces_of)
    v === nothing || (vec_sub[w] = v)
end

function morphology(families, vec_word, vec_sub, pieces_of)
    sims_word, sims_sub, shared = Float64[], Float64[], 0
    for (a, b) in families
        (haskey(vec_word, a) && haskey(vec_word, b)) || continue
        (haskey(vec_sub, a) && haskey(vec_sub, b)) || continue
        push!(sims_word, dot(vec_word[a], vec_word[b]))
        push!(sims_sub, dot(vec_sub[a], vec_sub[b]))
        isempty(intersect(pieces_of[a], pieces_of[b])) || (shared += 1)
    end
    sims_word, sims_sub, shared
end
sims_word, sims_sub, shared = morphology(families, vec_word, vec_sub, pieces_of)
@printf("  %d inflected pairs found (e.g. %s)\n", length(sims_word),
        join(("$a/$b" for (a, b) in families[1:min(4, end)]), ", "))
@printf("  they share a piece in %d of them (%.0f%%)\n", shared, 100shared / max(1, length(sims_word)))
@printf("  mean similarity, whole-word model  %+.3f\n", mean(sims_word))
@printf("  mean similarity, sub-word model    %+.3f\n", mean(sims_sub))

# ---------------------------------------------------------------- neighbours
println("\n=== nearest neighbours, both schemes ===\n")
probes = ["police", "fire", "government", "australia", "killed"]
wlist_w = collect(keys(vec_word)); wlist_s = collect(keys(vec_sub))
for p in probes
    haskey(counts, p) || continue
    @printf("%-11s whole-word  %s\n", p,
            join((@sprintf("%s (%.2f)", w, s) for (w, s) in neighbours_of(p, wlist_w, vec_word)), ", "))
    @printf("%-11s sub-word    %s\n", "",
            join((@sprintf("%s (%.2f)", w, s) for (w, s) in neighbours_of(p, wlist_s, vec_sub)), ", "))
end

println("\n=== words the whole-word model had to drop ===\n")
examples = sort([w for w in rare if haskey(vec_sub, w) && length(pieces_of[w]) > 1];
                by = w -> -counts[w])[1:min(5, end)]
for w in examples
    @printf("%-16s seen %d×, pieces %-28s -> %s\n", w, counts[w], join(pieces_of[w], "+"),
            join((@sprintf("%s (%.2f)", x, s) for (x, s) in neighbours_of(w, wlist_s, vec_sub; k = 3)), ", "))
end

# ---------------------------------------------------------------- figure
fig = Figure(size = (1080, 430))
ax1 = Axis(fig[1, 1]; title = "Which corpus words have a vector at all?", titlealign = :left,
           ylabel = "% of the $(length(all_words)) words", xticks = (1:2, ["whole word", "sub-word"]))
barplot!(ax1, 1:2, [100cov_word, 100cov_sub];
         color = [Makie.wong_colors()[1], Makie.wong_colors()[2]])
for (i, v) in enumerate([100cov_word, 100cov_sub])
    text!(ax1, i, v; text = @sprintf("%.1f%%", v), align = (:center, :bottom))
end
ylims!(ax1, 0, 108)

ax2 = Axis(fig[1, 2]; title = "Similarity of inflected pairs (e.g. attack / attacks)",
           titlealign = :left, xlabel = "cosine similarity", ylabel = "pairs")
hist!(ax2, sims_word; bins = 20, color = (Makie.wong_colors()[1], 0.6), label = "whole word")
hist!(ax2, sims_sub; bins = 20, color = (Makie.wong_colors()[2], 0.6), label = "sub-word")
axislegend(ax2; position = :lt, framevisible = false)
save(joinpath(figs, "subword_vs_word.png"), fig)
println("\nwrote ", joinpath(figs, "subword_vs_word.png"))
