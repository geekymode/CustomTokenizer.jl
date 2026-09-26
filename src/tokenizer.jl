# A tokenizer.json reader and BPE encoder, so the tokenizers real models use can
# be loaded and run from Julia.
#
# Every tokenizer examined here — GPT-2 (and so GPT-3), Gemma, Qwen — is byte
# pair encoding with an explicit vocabulary and merge list in the JSON. They
# differ in what happens around the merges: how text is normalized, how it is
# cut into chunks first, and what is done with characters the vocabulary does
# not contain.

"The GPT-2 byte ↔ printable character table, so every byte has a visible glyph."
const _BYTE_CHARS = let
    bs = collect(0x21:0x7e) ∪ collect(0xa1:0xac) ∪ collect(0xae:0xff)
    cs = Int.(bs)
    n = 0
    for b in 0x00:0xff
        if !(b in bs)
            push!(bs, b); push!(cs, 256 + n); n += 1
        end
    end
    Dict(bs[i] => Char(cs[i]) for i in eachindex(bs))
end
const _CHAR_BYTES = Dict(v => k for (k, v) in _BYTE_CHARS)

"The pre-tokenizer pattern GPT-2 uses when a ByteLevel component has no regex of its own."
const _GPT2_PATTERN = raw"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

"""
    BPETokenizer

A tokenizer loaded from a Hugging Face `tokenizer.json`: the vocabulary, the
ranked merge list, and the handful of flags that decide what happens around
them.

Built by [`load_hf_tokenizer`](@ref); used through [`encode`](@ref),
[`decode`](@ref) and [`token_strings`](@ref).

Ids are those in the file, counting from 0, so they can be compared directly
with any other implementation.
"""
struct BPETokenizer
    name::String
    vocab::Dict{String,Int}
    pieces::Vector{String}
    ranks::Dict{Tuple{String,String},Int}
    added::Vector{Pair{String,Int}}      # longest first
    bytelevel::Bool                      # bytes mapped to glyphs, GPT-2 style
    space_marker::Bool                   # spaces rewritten as ▁, SentencePiece style
    split_pattern::Union{Nothing,Regex}
    nfc::Bool
    byte_fallback::Bool
    ignore_merges::Bool
    unk::Union{Nothing,String}
end

function Base.show(io::IO, t::BPETokenizer)
    print(io, "BPETokenizer(", isempty(t.name) ? "" : t.name * ", ", length(t.vocab),
          " tokens, ", t.bytelevel ? "byte level" : "sentencepiece style", ")")
end

"""
    vocab_size(tokenizer) -> Int

How many entries the vocabulary has, added tokens included.
"""
vocab_size(t::BPETokenizer) = length(t.pieces)

"""
    token_string(tokenizer, id) -> Union{String,Nothing}

The raw piece for an id — `"▁queen"`, `"Ġcat"`, `"ization"` — or `nothing` if
the id is outside the vocabulary. Markers are left in place; use
[`decode`](@ref) for readable text.
"""
token_string(t::BPETokenizer, id::Integer) =
    (0 <= id < length(t.pieces)) ? t.pieces[id + 1] : nothing

"""
    token_id(tokenizer, piece) -> Union{Int,Nothing}

The id of an exact piece, or `nothing`. The piece must include its marker:
`token_id(gemma, "▁cat")`, not `"cat"`.
"""
token_id(t::BPETokenizer, piece::AbstractString) = get(t.vocab, String(piece), nothing)

"""
    load_hf_tokenizer(path; name="") -> BPETokenizer

Read a Hugging Face `tokenizer.json`. Combined with [`hf_download`](@ref) that
is the whole job:

```julia
gemma = load_hf_tokenizer(hf_download("google/gemma-4-E2B", "tokenizer.json"))
encode(gemma, "the cat drinks milk")
```

Byte-pair models are supported, which covers GPT-2 and GPT-3 (`r50k_base`),
Gemma, Qwen, Llama and most others. A WordPiece or Unigram file is rejected
rather than silently mis-encoded.
"""
function load_hf_tokenizer(path::AbstractString; name::AbstractString = "")
    j = JSON3.read(read(path, String))
    model = j["model"]
    mtype = haskey(model, :type) ? String(model["type"]) : "BPE"
    mtype == "BPE" || throw(ArgumentError("only BPE tokenizers are supported, this is $mtype"))

    vocab = Dict{String,Int}(String(k) => Int(v) for (k, v) in pairs(model["vocab"]))
    ranks = Dict{Tuple{String,String},Int}()
    for (i, m) in enumerate(model["merges"])
        pair = m isa AbstractString ?
               (parts = split(String(m), ' '; limit = 2); (parts[1], parts[2])) :
               (String(m[1]), String(m[2]))
        ranks[pair] = i
    end

    added = Pair{String,Int}[]
    if haskey(j, :added_tokens)
        for a in j["added_tokens"]
            push!(added, String(a["content"]) => Int(a["id"]))
            get!(vocab, String(a["content"]), Int(a["id"]))
        end
        sort!(added; by = p -> -length(first(p)))
    end

    pieces = fill("", maximum(values(vocab)) + 1)
    for (p, id) in vocab
        pieces[id + 1] = p
    end

    # what happens around the merges
    norm = get(j, :normalizer, nothing)
    space_marker = _has_component(norm, "Replace") &&
                   String(get(get(norm, :pattern, Dict()), :String, "")) == " "
    nfc = _has_component(norm, "NFC")

    pre = get(j, :pre_tokenizer, nothing)
    bytelevel = _has_component(pre, "ByteLevel")
    split_pattern = _split_regex(pre)
    if bytelevel && split_pattern === nothing
        split_pattern = Regex(_GPT2_PATTERN)
    end

    BPETokenizer(isempty(name) ? basename(dirname(path)) : String(name),
                 vocab, pieces, ranks, added, bytelevel, space_marker, split_pattern, nfc,
                 Bool(get(model, :byte_fallback, false)),
                 Bool(get(model, :ignore_merges, false)),
                 haskey(model, :unk_token) && model["unk_token"] !== nothing ?
                     String(model["unk_token"]) : nothing)
end

"Does this normalizer/pre-tokenizer contain a component of the given type?"
function _has_component(spec, wanted::AbstractString)
    spec === nothing && return false
    String(get(spec, :type, "")) == wanted && return true
    for key in (:pretokenizers, :normalizers)
        haskey(spec, key) && any(c -> _has_component(c, wanted), spec[key]) && return true
    end
    false
end

"The Split regex of a pre-tokenizer, if it has one."
function _split_regex(spec)
    spec === nothing && return nothing
    if String(get(spec, :type, "")) == "Split"
        pat = get(spec, :pattern, nothing)
        pat !== nothing && haskey(pat, :Regex) && return Regex(String(pat["Regex"]))
    end
    if haskey(spec, :pretokenizers)
        for c in spec[:pretokenizers]
            r = _split_regex(c)
            r === nothing || return r
        end
    end
    nothing
end

# ---------------------------------------------------------------- encoding
"Greedy byte-pair merging: repeatedly join the adjacent pair with the best rank."
function _merge(t::BPETokenizer, chunk::AbstractString)
    (t.ignore_merges && haskey(t.vocab, chunk)) && return [String(chunk)]
    syms = [string(c) for c in chunk]
    length(syms) < 2 && return syms
    while true
        best, at = typemax(Int), 0
        for i in 1:(length(syms) - 1)
            r = get(t.ranks, (syms[i], syms[i + 1]), typemax(Int))
            if r < best
                best, at = r, i
            end
        end
        at == 0 && break
        syms[at] *= syms[at + 1]
        deleteat!(syms, at + 1)
    end
    syms
end

"Map one merged symbol to ids, falling back to raw bytes when it is unknown."
function _ids_for(t::BPETokenizer, sym::AbstractString, out::Vector{Int})
    id = get(t.vocab, sym, nothing)
    if id !== nothing
        push!(out, id)
    elseif t.byte_fallback
        for b in codeunits(sym)
            bid = get(t.vocab, "<0x" * uppercase(string(b; base = 16, pad = 2)) * ">", nothing)
            bid === nothing || push!(out, bid)
        end
    elseif t.unk !== nothing && haskey(t.vocab, t.unk)
        push!(out, t.vocab[t.unk])
    end
    out
end

"Split off any added tokens, which are matched literally and never merged."
function _segments(t::BPETokenizer, text::AbstractString)
    isempty(t.added) && return Any[text]
    out = Any[]
    rest = String(text)
    while !isempty(rest)
        hit, at = nothing, 0
        for (content, id) in t.added
            r = findfirst(content, rest)
            if r !== nothing && (at == 0 || first(r) < at)
                hit, at = (content => id, first(r))
            end
        end
        hit === nothing && (push!(out, rest); break)
        at > 1 && push!(out, rest[1:prevind(rest, at)])
        push!(out, hit)
        rest = rest[(at + ncodeunits(first(hit))):end]
    end
    out
end

"""
    encode(tokenizer, text) -> Vector{Int}

Token ids for `text`, matching the reference implementation exactly (the test
suite checks this against recorded output from Hugging Face's Rust tokenizers).

Special tokens are not added: what you pass is what is encoded. A special token
appearing literally in the text *is* matched, so a user cannot forge one by
typing it.
"""
function encode(t::BPETokenizer, text::AbstractString)
    ids = Int[]
    for seg in _segments(t, text)
        if seg isa Pair
            push!(ids, last(seg))
            continue
        end
        s = String(seg)
        isempty(s) && continue
        t.nfc && (s = Unicode.normalize(s, :NFC))
        t.space_marker && (s = replace(s, ' ' => '▁'))
        chunks = t.split_pattern === nothing ? [s] :
                 [m.match for m in eachmatch(t.split_pattern, s)]
        for chunk in chunks
            isempty(chunk) && continue
            mapped = t.bytelevel ? String([_BYTE_CHARS[b] for b in codeunits(chunk)]) : String(chunk)
            for sym in _merge(t, mapped)
                _ids_for(t, sym, ids)
            end
        end
    end
    ids
end

"""
    token_strings(tokenizer, text) -> Vector{String}

The pieces `text` is cut into, with their markers: `["the", "▁cat"]` for Gemma,
`["the", "Ġcat"]` for a byte-level tokenizer.
"""
token_strings(t::BPETokenizer, text::AbstractString) =
    String[something(token_string(t, id), "<$id>") for id in encode(t, text)]

# ---------------------------------------------------------------- decoding
"""
    decode(tokenizer, ids) -> String

Turn ids back into text. Byte-level tokenizers round-trip exactly; marker-based
ones restore spaces from `▁` and reassemble `<0xNN>` byte tokens, which is how
characters outside the vocabulary survive.
"""
function decode(t::BPETokenizer, ids::AbstractVector{<:Integer})
    bytes = UInt8[]
    for id in ids
        piece = token_string(t, id)
        piece === nothing && continue
        if t.bytelevel
            for c in piece
                b = get(_CHAR_BYTES, c, nothing)
                b === nothing || push!(bytes, b)
            end
        else
            m = match(r"^<0x([0-9A-Fa-f]{2})>$", piece)
            if m !== nothing
                push!(bytes, parse(UInt8, m.captures[1]; base = 16))
            else
                append!(bytes, codeunits(replace(piece, '▁' => ' ')))
            end
        end
    end
    String(bytes)
end

decode(t::BPETokenizer, text::AbstractString) = decode(t, encode(t, text))
