# Compare vectors trained here with published ones.
#
#   python experiments/fetch_pretrained.py --words 50000 --dim 50
#   julia --project=.. examples/analogies.jl ../text8 17000000   # writes ours-text8.vec
#   julia --project=.. examples/compare_pretrained.jl
#
# The two sets of vectors live in unrelated coordinate systems — the numbers in
# one say nothing about the numbers in the other. Everything below compares
# structure instead: who is near whom, and how pairs are ranked.

using CustomTokenizer, Printf, Random, Statistics

results = joinpath(@__DIR__, "..", "experiments", "results")
ours_path  = joinpath(results, "ours-text8.vec")
glove_path = joinpath(results, "glove-50d-50k.vec")

for (what, path) in (("ours", ours_path), ("GloVe", glove_path))
    isfile(path) || error("$what vectors not found at $path — see the header of this file")
end

ours  = load_vectors(ours_path;  name = "ours (text8, 17M words)")
glove = load_vectors(glove_path; name = "GloVe (6B words)")

@printf("%-26s %6d words × %d dims\n", ours.name, length(ours.vocab), size(ours.W, 1))
@printf("%-26s %6d words × %d dims\n", glove.name, length(glove.vocab), size(glove.W, 1))

shared = shared_vocabulary(ours, glove)
@printf("shared vocabulary: %d words (%.0f%% of ours)\n\n",
        length(shared), 100length(shared) / length(ours.vocab))

# ---------------------------------------------------------------- side by side
println("=== nearest neighbours, side by side ===\n")
for w in ("king", "paris", "water", "aunt", "computer", "war")
    (haskey(ours.vocab, w) && haskey(glove.vocab, w)) || continue
    fmt(e) = join((@sprintf("%s (%.2f)", x, s) for (x, s) in nearest_neighbours(e, w, 5)), ", ")
    @printf("%-9s ours   %s\n", w, fmt(ours))
    @printf("%-9s GloVe  %s\n\n", "", fmt(glove))
end

# ---------------------------------------------------------------- agreement
println("=== how much do they agree? ===\n")
probe = shuffle(Xoshiro(7), shared)[1:min(500, length(shared))]
ov = neighbour_overlap(ours, glove, probe; k = 10)
@printf("  neighbour overlap @10, over %d shared words: %.1f%%\n", length(ov.per_word), 100ov.mean)

best = sort(collect(ov.per_word); by = x -> -x[2])[1:5]
worst = sort(collect(ov.per_word); by = x -> x[2])[1:5]
@printf("    most agreement:  %s\n", join((@sprintf("%s %.0f%%", w, 100v) for (w, v) in best), ", "))
@printf("    least agreement: %s\n", join((@sprintf("%s %.0f%%", w, 100v) for (w, v) in worst), ", "))

agree = similarity_agreement(ours, glove; pairs = 4000)
@printf("  similarity correlation over %d random pairs: %.3f\n\n", agree.n, agree.correlation)

# ---------------------------------------------------------------- analogies
println("=== analogies ===\n")
for e in (ours, glove)
    acc = analogy_accuracy(e; k = 1)
    acc4 = analogy_accuracy(e; k = 4)
    @printf("%-26s %d/%d at rank 1, %d/%d in the top 4",
            e.name, acc.hits, acc.asked, acc4.hits, acc4.asked)
    isempty(acc.skipped) || @printf("  (skipped %d: missing words)", length(acc.skipped))
    println()
    for (a, b, c, want, got, ok) in acc4.results
        @printf("    %s %-7s → %-8s as %-8s → %-9s got %s\n",
                ok ? "✓" : " ", a, b, c, want, join(got, ", "))
    end
    println()
end

println("""
Reading this: the two models were trained on different corpora (17M words
against 6B) with different algorithms, so disagreement is expected. What
matters is whether ours is wrong in a *systematic* way — a low overlap on
common nouns would mean undertraining, while disagreement only on rare or
ambiguous words is the corpus talking.""")
