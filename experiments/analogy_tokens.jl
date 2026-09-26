# Do the analogy words survive as single tokens?
#
#   julia --project=. experiments/analogy_tokens.jl
#
# A tokenizer has no vectors, so "man - woman + queen" cannot be computed in
# one. What it decides is whether a word is one unit or several, and that
# decides whether the model behind it has a single vector for that word at all.

using CustomTokenizer, Printf

REPOS = [("GPT-3", "gpt2"), ("Gemma 4", "google/gemma-4-E2B"),
         ("Gemma 2", "philschmid/gemma-tokenizer-chatml"), ("Qwen 2.5", "Qwen/Qwen2.5-0.5B")]

QUESTIONS = [("man", "king", "woman", "queen"), ("man", "uncle", "woman", "aunt"),
             ("boy", "son", "girl", "daughter"), ("france", "paris", "italy", "rome"),
             ("germany", "berlin", "japan", "tokyo"), ("good", "better", "bad", "worse"),
             ("walk", "walking", "swim", "swimming")]

tokenizers = Pair{String,BPETokenizer}[]
for (name, repo) in REPOS
    try
        push!(tokenizers, name => load_hf_tokenizer(hf_download(repo, "tokenizer.json"); name = name))
    catch err
        @warn "skipping $name" repo
    end
end

words = sort(unique(collect(Iterators.flatten(QUESTIONS))))
pieces_of(t, w) = token_strings(t, " " * w)

println("\n=== each analogy word, with its leading space ===\n")
@printf("  %-10s", "word"); foreach(p -> @printf("%-22s", first(p)), tokenizers); println()
println("  ", "-"^(10 + 22length(tokenizers)))
singles = Dict(name => 0 for (name, _) in tokenizers)
for w in words
    @printf("  %-10s", w)
    for (name, t) in tokenizers
        ps = pieces_of(t, w)
        length(ps) == 1 && (singles[name] += 1)
        @printf("%-22s", join(strip.(replace.(ps, "▁" => "", "Ġ" => "")), "|"))
    end
    println()
end

println("\n=== single-token words ===\n")
for (name, _) in tokenizers
    @printf("  %-10s %2d/%d\n", name, singles[name], length(words))
end

println("\n=== questions where every word is a single token ===\n")
for (a, b, c, want) in QUESTIONS
    @printf("  %-28s", "$a/$b/$c/$want")
    for (name, t) in tokenizers
        ok = all(w -> length(pieces_of(t, w)) == 1, (a, b, c, want))
        @printf("  %s: %-4s", name, ok ? "yes" : "no")
    end
    println()
end

println("""

Only for those questions does the model behind the tokenizer have one input
vector per word. Everywhere else the word is a sequence, and an analogy over it
has to pool several vectors — a different operation with a different failure
mode.""")
