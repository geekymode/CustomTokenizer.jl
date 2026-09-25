# The whole pipeline, printed. Mirrors the "Getting started" page.
#
# Run:  julia --project=.. walkthrough.jl

using CustomTokenizer, Printf

section(t) = println("\n", "="^70, "\n", t, "\n", "="^70)

section("1-2. text → words → ids")
sentences = tokenize(TOY_TEXT)
vocab = build_vocab(sentences)
@printf("%d sentences, %d words, %d distinct\n", length(sentences), sum(length, sentences), length(vocab))
show(stdout, MIME"text/plain"(), vocab); println()

section("3. two tables")
model = Model(vocab; dim = 8, window = 2, negative = 3)
show(stdout, MIME"text/plain"(), model)
println("  W[:, cat] = ", round.(wordvec(model, "cat"); digits = 4))
println("  C[:, cat] = ", round.(ctxvec(model, "cat"); digits = 4), "   (all zero)")

section("4. training pairs")
pairs = corpus_pairs(sentences, vocab; window = 2)
N = cooccurrence(sentences, vocab; window = 2)
@printf("%d pairs; the table is symmetric: %s; it sums to %d\n", length(pairs), N == N', sum(N))
println("cat's neighbours: ",
        [(vocab[j], N[vocab["cat"], j]) for j in 1:length(vocab) if N[vocab["cat"], j] > 0])

section("5. one update, traced")
info = update_pair!(model, "cat", "drinks", ["on", "wears", "queen"], 0.5)
show(stdout, MIME"text/plain"(), info)
println("  C[:, drinks] is now 0.25 × W[:, cat]: ",
        ctxvec(model, "drinks") ≈ 0.25 .* wordvec(model, "cat"))

section("6. the full run")
trainlog = train!(model, pairs; epochs = 300, snapshots = [0, 10, 50, 300])
for e in (1, 10, 50, 100, 300)
    @printf("  epoch %3d   loss %.3f\n", e, trainlog.losses[e])
end

section("7. what came out")
for (a, b) in (("king", "queen"), ("cat", "dog"), ("milk", "water"), ("cat", "king"))
    @printf("  %-6s ~ %-6s %+.2f\n", a, b, similarity(model, a, b))
end
println()
for w in ("cat", "king", "milk", "castle")
    @printf("  %-7s -> %s\n", w,
            join((@sprintf("%s (%.2f)", x, s) for (x, s) in nearest_neighbours(model, w, 3)), ", "))
end

section("does it match what the counts predict?")
probs = sampling_probabilities(UniformNegatives(), length(vocab))
for (a, b) in (("cat", "drinks"), ("cat", "chases"), ("cat", "crown"))
    tgt = target_probability(N, row_totals(N), probs, vocab[a], vocab[b], 3)
    @printf("  P(%-7s | %-4s): target %.2f, trained %.2f\n", b, a, tgt, probability(model, a, b))
end
c = calibration(model, N, probs)
@printf("  across all %d pairs: correlation %.3f, mean gap %.3f\n",
        c.n, c.correlation, c.mean_absolute_error)
