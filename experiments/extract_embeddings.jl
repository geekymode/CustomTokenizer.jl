# Pull an LLM's input embedding table out of its checkpoint — in Julia.
#
#   julia --project=. experiments/extract_embeddings.jl                      # Qwen 2.5 0.5B
#   julia --project=. experiments/extract_embeddings.jl gpt2                 # any repo
#   julia --project=. experiments/extract_embeddings.jl Qwen/Qwen2.5-0.5B 30000
#
# The first thing a transformer does with a token id is look up a row of its
# embedding table. Those rows are word vectors of a sort, and once written out
# they can be compared with anything else this package loads.
#
# Downloads the checkpoint (hundreds of MB to a few GB). No Python, no torch:
# safetensors is a JSON header followed by raw bytes.

using CustomTokenizer, Printf

repo = length(ARGS) >= 1 ? ARGS[1] : "Qwen/Qwen2.5-0.5B"
maxwords = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30_000
slug = replace(lowercase(split(repo, "/")[end]), r"[^a-z0-9]+" => "")

println("tokenizer ...")
tok = load_hf_tokenizer(hf_download(repo, "tokenizer.json"); name = repo)
println("  ", tok)

println("weights (this is the big download) ...")
weights = hf_download(repo, "model.safetensors")
@printf("  %s, %.0f MB\n", basename(weights), filesize(weights) / 1e6)
@printf("  tensors: %s …\n", join(first(safetensors_names(weights), 2), ", "))

emb = embedding_from_checkpoint(weights, tok; max_words = maxwords)
@printf("\nkept %d whole-word rows of %d numbers each\n", length(emb.vocab), size(emb.W, 1))

println("\nnearest neighbours:")
for w in ("king", "water", "paris", "computer")
    haskey(emb.vocab, w) || continue
    @printf("  %-9s %s\n", w,
            join((@sprintf("%s (%.2f)", x, s) for (x, s) in nearest_neighbours(emb, w, 5)), ", "))
end

acc = analogy_accuracy(emb)
@printf("\nanalogies: %d/%d at rank 1", acc.hits, acc.asked)
isempty(acc.skipped) || @printf("  (%d skipped, words missing)", length(acc.skipped))
println()
for (a, b, c, want, got, ok) in analogy_accuracy(emb; k = 4).results
    @printf("  %s %-7s → %-8s as %-8s → %-9s got %s\n", ok ? "✓" : " ", a, b, c, want,
            join(got, ", "))
end

out = joinpath(@__DIR__, "results", slug * "-embeddings.vec")
mkpath(dirname(out))
save_vectors(out, emb)
@printf("\nwrote %s (%.0f MB)\n", out, filesize(out) / 1e6)
println("compare it with the rest:  julia --project=. examples/compare_llm_embeddings.jl")
