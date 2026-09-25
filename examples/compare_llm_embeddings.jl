# Vectors trained here, vectors downloaded, and an LLM's own input embeddings —
# all three answering the same questions.
#
# Prepare whichever you have (each is optional; the script uses what it finds):
#
#   julia --project=. examples/analogies.jl ../text8 17000000      # ours-text8.vec
#   python experiments/fetch_pretrained.py --words 50000 --dim 50  # glove-50d-50k.vec
#   python experiments/extract_llm_embeddings.py                   # gemma-3-270m-embeddings.vec
#
# then:
#
#   julia --project=. examples/compare_llm_embeddings.jl

using CustomTokenizer, Printf, Random, Statistics

results = joinpath(@__DIR__, "..", "experiments", "results")

candidates = [
    ("ours (text8)",        joinpath(results, "ours-text8.vec")),
    ("GloVe 6B 50d",        joinpath(results, "glove-50d-50k.vec")),
    ("Gemma 3 270M inputs", joinpath(results, "gemma-3-270m-embeddings.vec")),
    ("Qwen 2.5 0.5B inputs", joinpath(results, "Qwen2.5-0.5B-embeddings.vec")),
]

embeddings = Pair{String,Embedding}[]
for (name, path) in candidates
    if isfile(path)
        push!(embeddings, name => load_vectors(path; name = name))
    else
        @info "skipping $name — not found at $(basename(path)); see the header of this file"
    end
end
isempty(embeddings) && error("nothing to compare: prepare at least one vector file")

println("\n=== what we have ===\n")
for (name, e) in embeddings
    @printf("  %-22s %6d words × %3d dims\n", name, length(e.vocab), size(e.W, 1))
end

# ---------------------------------------------------------------- analogies
println("\n=== analogies: ", length(ANALOGY_QUESTIONS), " questions ===\n")
for (name, e) in embeddings
    one = analogy_accuracy(e; k = 1)
    four = analogy_accuracy(e; k = 4)
    @printf("  %-22s %d/%d at rank 1, %d/%d in the top 4", name, one.hits, one.asked,
            four.hits, four.asked)
    isempty(one.skipped) || @printf("  (%d skipped: words missing)", length(one.skipped))
    println()
end

println("\n  where they differ:\n")
for (a, b, c, want) in ANALOGY_QUESTIONS
    line = @sprintf("    %-7s → %-8s as %-8s → %-9s", a, b, c, want)
    any_asked = false
    for (name, e) in embeddings
        all(w -> haskey(e.vocab, w), (a, b, c, want)) || continue
        any_asked = true
        got = first(first.(analogy(e, a, b, c; k = 1)))
        line *= @sprintf("  %s: %-12s", split(name)[1], got == want ? "✓" : got)
    end
    any_asked && println(line)
end

# ---------------------------------------------------------------- agreement
length(embeddings) > 1 && println("\n=== do they agree with each other? ===\n")
for i in 1:length(embeddings), j in (i + 1):length(embeddings)
    (na, a), (nb, b) = embeddings[i], embeddings[j]
    shared = shared_vocabulary(a, b)
    length(shared) < 50 && continue
    probe = shuffle(Xoshiro(7), shared)[1:min(300, length(shared))]
    ov = neighbour_overlap(a, b, probe; k = 10)
    ag = similarity_agreement(a, b; pairs = 3000)
    @printf("  %-22s vs %-22s  %5d shared · overlap@10 %4.1f%% · correlation %+.3f\n",
            na, nb, length(shared), 100ov.mean, ag.correlation)
end

# ---------------------------------------------------------------- neighbours
println("\n=== nearest neighbours, side by side ===\n")
for w in ("king", "water", "paris", "computer")
    any(e -> haskey(e.second.vocab, w), embeddings) || continue
    println("  ", w)
    for (name, e) in embeddings
        haskey(e.vocab, w) || continue
        @printf("    %-22s %s\n", name,
                join((@sprintf("%s (%.2f)", x, s) for (x, s) in nearest_neighbours(e, w, 5)), ", "))
    end
    println()
end

println("""
A note on reading this: an LLM's input embeddings are not trained to be good
word vectors. They are the first layer of a stack that adds context at every
step afterwards, so a word's meaning in that model is spread across the whole
network rather than living in its row. Expect them to look worse on analogies
than GloVe, which is trained for exactly this — and treat that as information
about what the row does, not as a defect.""")
