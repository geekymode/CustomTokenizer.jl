# Reading model checkpoints, so an LLM's embedding table can be compared with
# vectors trained here. A safetensors file is a little-endian UInt64 giving the
# length of a JSON header, the header itself, then the raw tensor bytes — which
# is simple enough to read directly.

"""
    safetensors_header(path) -> (header, data_offset)

The JSON header of a safetensors file and the byte offset where tensor data
begins. Each entry gives a tensor's `dtype`, `shape` and `data_offsets`.
"""
function safetensors_header(path::AbstractString)
    open(path) do io
        n = read(io, UInt64)
        header = JSON3.read(String(read(io, n)))
        (header, 8 + Int(n))
    end
end

"""
    safetensors_names(path) -> Vector{String}

Every tensor in the file, sorted. Useful for finding what a checkpoint calls
its embedding table.
"""
function safetensors_names(path::AbstractString)
    header, _ = safetensors_header(path)
    sort([String(k) for k in keys(header) if String(k) != "__metadata__"])
end

"""
    read_safetensor(path, name) -> Matrix{Float32}

One tensor, as `Float32`. `F32`, `F16` and `BF16` are handled; bfloat16 is
widened by moving each 16-bit value into the top half of a float32, which is
exactly what bfloat16 is.

Tensors are stored row-major, so the result is transposed into Julia's
column-major layout: a `(vocab, dim)` checkpoint tensor comes back `dim × vocab`,
one column per token, matching [`Embedding`](@ref).
"""
function read_safetensor(path::AbstractString, name::AbstractString)
    header, base = safetensors_header(path)
    haskey(header, Symbol(name)) ||
        throw(KeyError("$name not in $(basename(path)); has $(first(safetensors_names(path), 4))…"))
    meta = header[Symbol(name)]
    lo, hi = meta["data_offsets"]
    shape = Int.(meta["shape"])
    raw = open(path) do io
        seek(io, base + lo)
        read(io, hi - lo)
    end
    dtype = String(meta["dtype"])
    flat = if dtype == "BF16"
        reinterpret(Float32, UInt32.(reinterpret(UInt16, raw)) .<< 16)
    elseif dtype == "F16"
        Float32.(reinterpret(Float16, raw))
    elseif dtype == "F32"
        copy(reinterpret(Float32, raw))
    else
        throw(ArgumentError("dtype $dtype is not handled (F32, F16 and BF16 are)"))
    end
    length(shape) == 2 || throw(ArgumentError("expected a matrix, got shape $shape"))
    # row-major (rows, cols) -> column-major cols × rows
    permutedims(reshape(flat, reverse(shape)...), (1, 2))
end

"""
    hf_download(repo, file; revision="main", dir=nothing) -> String

Download one file from a Hugging Face repository and return the local path,
skipping the download if it is already there. No Python, no authentication —
just the public resolve URL.

```julia
path = hf_download("google/gemma-4-E2B", "tokenizer.json")
tok  = load_hf_tokenizer(path)
```
"""
function hf_download(repo::AbstractString, file::AbstractString;
                     revision::AbstractString = "main", dir = nothing)
    root = dir === nothing ? joinpath(get(ENV, "HOME", tempdir()), ".cache",
                                      "CustomTokenizer", replace(repo, "/" => "--")) : dir
    mkpath(root)
    dest = joinpath(root, basename(file))
    isfile(dest) && return dest
    url = "https://huggingface.co/$repo/resolve/$revision/$file"
    @info "downloading" url
    Downloads.download(url, dest)
    dest
end

"""
    embedding_from_checkpoint(weights, tokenizer; max_words=50_000, tensor="") -> Embedding

Pull an LLM's input embedding table out of a safetensors checkpoint and keep the
rows whose token is a whole word, so they can be compared with word vectors.

`▁queen` and `Ġqueen` become `queen`; word pieces and punctuation are dropped,
since they are not words. The word-initial form of a token wins over a
continuation form of the same spelling.

    julia> tok = load_hf_tokenizer(hf_download("Qwen/Qwen2.5-0.5B", "tokenizer.json"))
    julia> emb = embedding_from_checkpoint(hf_download("Qwen/Qwen2.5-0.5B", "model.safetensors"), tok)
    julia> analogy(emb, "man", "king", "woman")

These rows are not trained to be standalone word vectors — they feed a stack
that adds context at every layer — so it is tempting to assume they must be
poor ones. Measured on the built-in questions, Qwen 2.5 0.5B scores 11/12,
against 9/12 for GloVe and 7/12 for a word2vec run on text8. Worth checking
rather than assuming.
"""
function embedding_from_checkpoint(weights::AbstractString, tokenizer;
                                   max_words::Integer = 50_000, tensor::AbstractString = "")
    candidates = ("model.embed_tokens.weight", "embed_tokens.weight",
                  "transformer.wte.weight", "wte.weight", "tok_embeddings.weight")
    names = safetensors_names(weights)
    key = isempty(tensor) ? something(findfirst(in(names), collect(candidates)), 0) : 0
    name = isempty(tensor) ?
           (key == 0 ? throw(ArgumentError("no embedding tensor in $(basename(weights)); " *
                                           "saw $(first(names, 5))… — pass tensor=")) :
            candidates[key]) : tensor
    E = read_safetensor(weights, name)                 # dim × vocab
    words = String[]
    cols = Int[]
    seen = Set{String}()
    for id in 0:(min(size(E, 2), vocab_size(tokenizer)) - 1)
        piece = token_string(tokenizer, id)
        piece === nothing && continue
        initial = startswith(piece, '▁') || startswith(piece, 'Ġ')
        word = initial ? piece[nextind(piece, 1):end] : piece
        (length(word) > 1 && all(c -> isletter(c) || c in "-'", word)) || continue
        lw = lowercase(word)
        (lw in seen && !initial) && continue
        lw in seen && continue
        push!(seen, lw); push!(words, lw); push!(cols, id + 1)
        length(words) >= max_words && break
    end
    vocab = Vocabulary(words, Dict(w => i for (i, w) in enumerate(words)), fill(0, length(words)))
    Embedding(vocab, Float64.(E[:, cols]), basename(dirname(weights)))
end
