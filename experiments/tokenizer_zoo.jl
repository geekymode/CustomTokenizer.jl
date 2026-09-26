# The tokenizers real models use, compared — in Julia, with no Python.
#
#   julia --project=. experiments/tokenizer_zoo.jl
#
# Downloads only tokenizer.json files (a few MB each), never model weights.
# Already-downloaded files are reused, so a second run needs no network.

using CustomTokenizer, Printf

REPOS = [
    ("GPT-3 / GPT-2", "gpt2"),                 # r50k_base
    ("Gemma 4",       "google/gemma-4-E2B"),
    ("Gemma 2",       "philschmid/gemma-tokenizer-chatml"),
    ("Qwen 2.5",      "Qwen/Qwen2.5-0.5B"),
]

TEXTS = [
    "english"  => "The cat drinks milk while the queen wears a crown in her castle.",
    "german"   => "Die Katze trinkt Milch, während die Königin eine Krone trägt.",
    "hindi"    => "बिल्ली दूध पीती है जबकि रानी अपने महल में मुकुट पहनती है।",
    "chinese"  => "猫在喝牛奶，女王在她的城堡里戴着王冠。",
    "code"     => "def total(xs):\n    \"\"\"Sum a list.\"\"\"\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
    "numbers"  => "In 2026 the price rose from 1234.56 to 98765.43, a 7900% change.",
    "emoji"    => "the cat 🐈 drinks milk 🥛 — naïve café façade 🫩",
]

PROBES = [
    "a year"          => "2026",
    "a long number"   => "1234567",
    "indentation"     => "def f():\n    return 1",
    "repeated spaces" => "a     b",
    "an emoji"        => "🫩",
    "capitalisation"  => "Cat cat CAT",
    "a rare word"     => "antidisestablishmentarianism",
]

section(t) = println("\n", "="^78, "\n", t, "\n", "="^78, "\n")

tokenizers = Pair{String,BPETokenizer}[]
for (name, repo) in REPOS
    try
        push!(tokenizers, name => load_hf_tokenizer(hf_download(repo, "tokenizer.json"); name = name))
    catch err
        @warn "skipping $name" repo exception = (err, catch_backtrace()[1:1])
    end
end
isempty(tokenizers) && error("no tokenizers available — is huggingface.co reachable?")

section("the tokenizers")
@printf("  %-16s %10s  %s\n", "tokenizer", "vocabulary", "kind")
for (name, t) in tokenizers
    @printf("  %-16s %10d  %s\n", name, vocab_size(t),
            t.bytelevel ? "byte level BPE" : "sentencepiece style BPE" *
            (t.byte_fallback ? ", byte fallback" : ""))
end

section("tokens per text (fewer is cheaper, and leaves more context)")
@printf("  %-9s %6s %6s", "text", "bytes", "words")
foreach(p -> @printf("%16s", first(p)), tokenizers); println()
for (label, text) in TEXTS
    @printf("  %-9s %6d %6d", label, ncodeunits(text), length(split(text)))
    for (_, t) in tokenizers
        @printf("%16d", length(encode(t, text)))
    end
    println()
end

section("bytes per token (higher packs more text into the same window)")
@printf("  %-9s", "text")
foreach(p -> @printf("%16s", first(p)), tokenizers); println()
for (label, text) in TEXTS
    @printf("  %-9s", label)
    for (_, t) in tokenizers
        @printf("%16.2f", ncodeunits(text) / max(1, length(encode(t, text))))
    end
    println()
end

section("how each one cuts things up")
for (label, probe) in PROBES
    println("  ", label, ":  ", repr(probe))
    for (name, t) in tokenizers
        pieces = token_strings(t, probe)
        @printf("    %-16s %3d  %s\n", name, length(pieces), join(pieces, " | "))
    end
    println()
end

section("decode(encode(x)) == x ?")
for (name, t) in tokenizers
    ok = count(((_, text),) -> decode(t, encode(t, text)) == text, TEXTS)
    @printf("  %-16s %d/%d exact\n", name, ok, length(TEXTS))
end
