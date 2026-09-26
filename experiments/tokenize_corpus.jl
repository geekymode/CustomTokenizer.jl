# Tokenize one corpus two ways, so the same text can be trained on twice.
#
#   julia --project=. experiments/tokenize_corpus.jl ../../lee_background.cor
#   julia --project=. experiments/tokenize_corpus.jl corpus.txt 200000     # char limit
#
# Writes into experiments/results/:
#
#   stream_word.txt    one sentence per line, this package's rule (lowercase,
#                      letters only)
#   stream_gemma.txt   the same sentences as Gemma 4 sub-word pieces
#   word_pieces.tsv    word -> the pieces it is made of, so a word vector can be
#                      built by pooling its pieces
#   tokenize_stats.tsv counts for both streams
#
# Only the tokenizer file is downloaded (a few MB); no model weights.

using CustomTokenizer, Printf

corpus = length(ARGS) >= 1 ? ARGS[1] : error("usage: tokenize_corpus.jl CORPUS [CHAR_LIMIT]")
limit  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
isfile(corpus) || error("corpus not found: $corpus")

const GEMMA = "google/gemma-4-E2B"
const SENTENCE = r"(?<=[.!?])\s+"
const WORD = r"[A-Za-z]+"

results = joinpath(@__DIR__, "results"); mkpath(results)

text = read(corpus, String)
limit > 0 && (text = text[1:min(limit, lastindex(text))])
sentences = String[]
for block in eachsplit(text, '\n'), s in eachsplit(block, SENTENCE)
    s = strip(s)
    length(s) > 1 && push!(sentences, String(s))
end
@printf("read %d sentences from %s\n", length(sentences), basename(corpus))

tok = load_hf_tokenizer(hf_download(GEMMA, "tokenizer.json"); name = GEMMA)
println("tokenizer: ", tok)

"""
    build_streams(tok, sentences)

Both token streams in one pass: whole words by this package's rule, and Gemma
pieces from the same sentence lowercased, so the unit is the only difference.
"""
function build_streams(tok, sentences)
    word_lines, gemma_lines = String[], String[]
    vocab_words, vocab_pieces = Set{String}(), Set{String}()
    n_word = n_gemma = 0

    for s in sentences
        words = [lowercase(m.match) for m in eachmatch(WORD, s)]
        length(words) < 2 && continue
        push!(word_lines, join(words, " "))
        union!(vocab_words, words)
        n_word += length(words)

        pieces = [p for p in token_strings(tok, lowercase(s)) if !isempty(strip(p))]
        push!(gemma_lines, join(pieces, " "))
        union!(vocab_pieces, pieces)
        n_gemma += length(pieces)
    end
    (; word_lines, gemma_lines, vocab_words, vocab_pieces, n_word, n_gemma)
end

st = build_streams(tok, sentences)
word_lines, gemma_lines = st.word_lines, st.gemma_lines
vocab_words, vocab_pieces = st.vocab_words, st.vocab_pieces
n_word, n_gemma = st.n_word, st.n_gemma

write(joinpath(results, "stream_word.txt"), join(word_lines, "\n") * "\n")
write(joinpath(results, "stream_gemma.txt"), join(gemma_lines, "\n") * "\n")

# how each word breaks into pieces (word-initial form, i.e. preceded by a space)
function write_pieces(path, tok, vocab_words)
    whole = 0
    open(path, "w") do f
        for w in sort(collect(vocab_words))
            ps = [p for p in token_strings(tok, " " * w) if !isempty(strip(p))]
            length(ps) == 1 && (whole += 1)
            println(f, w, '\t', join(ps, " "))
        end
    end
    whole
end
whole = write_pieces(joinpath(results, "word_pieces.tsv"), tok, vocab_words)

open(joinpath(results, "tokenize_stats.tsv"), "w") do f
    println(f, "measure\tword\tgemma")
    println(f, "sentences\t", length(word_lines), '\t', length(gemma_lines))
    println(f, "tokens\t", n_word, '\t', n_gemma)
    println(f, "distinct\t", length(vocab_words), '\t', length(vocab_pieces))
end

@printf("\nsentences            %d\n", length(word_lines))
@printf("tokens   word-level  %d\n", n_word)
@printf("         sub-word    %d  (%.2f× as many)\n", n_gemma, n_gemma / n_word)
@printf("distinct word-level  %d\n", length(vocab_words))
@printf("         sub-word    %d pieces\n", length(vocab_pieces))
@printf("whole words kept as one piece: %d/%d (%.0f%%)\n", whole, length(vocab_words),
        100whole / length(vocab_words))
@printf("\nwrote four files to %s\n", results)
