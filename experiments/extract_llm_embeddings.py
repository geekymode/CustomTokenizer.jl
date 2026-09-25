"""Pull the input embedding matrix out of an open-weights LLM.

A tokenizer has no vectors, but the model behind it does: the first thing any
transformer does is look up a row of its embedding table for each token id.
Those rows are word vectors of a sort, and this script writes them in the same
plain-text format CustomTokenizer.load_vectors reads, so they can be compared
with vectors trained here or with GloVe.

    python experiments/extract_llm_embeddings.py                      # Gemma 3 270M
    python experiments/extract_llm_embeddings.py --repo Qwen/Qwen2.5-0.5B
    python experiments/extract_llm_embeddings.py --self-test          # no download

Only rows whose token is a whole word are kept: a token like "▁queen" becomes
the word "queen", while word pieces ("▁anti", "ization") and punctuation are
skipped, since they are not words and cannot be compared with word vectors.

Weights are read with a small safetensors reader below, so neither torch nor
the safetensors package is needed — numpy is enough. bfloat16 is converted by
widening each 16-bit value into the top half of a float32, which is exactly
what bfloat16 is.
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
EMBED_KEYS = ("model.embed_tokens.weight", "embed_tokens.weight",
              "transformer.wte.weight", "wte.weight", "tok_embeddings.weight")
WORD = re.compile(r"^[A-Za-z][A-Za-z\-']*$")


# ---------------------------------------------------------------- safetensors
def read_header(path):
    """(header dict, offset where the data starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_tensor(path, name):
    """One tensor from a safetensors file, as float32."""
    header, base = read_header(path)
    if name not in header:
        raise KeyError(f"{name} not in {path.name}; has {list(header)[:5]} ...")
    meta = header[name]
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + start)
        raw = f.read(end - start)
    dtype = meta["dtype"]
    if dtype == "BF16":
        # bfloat16 is the top 16 bits of a float32
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        flat = u16.view(np.float32)
    elif dtype in ("F16", "FP16"):
        flat = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype in ("F32", "FP32"):
        flat = np.frombuffer(raw, dtype="<f4")
    else:
        raise NotImplementedError(f"dtype {dtype} not handled")
    return flat.reshape(meta["shape"])


def find_embeddings(repo):
    """Download whichever shard holds the embedding table, and read it."""
    from huggingface_hub import hf_hub_download

    try:                                   # sharded checkpoints have an index
        index = Path(hf_hub_download(repo_id=repo, filename="model.safetensors.index.json"))
        weight_map = json.loads(index.read_text())["weight_map"]
        key = next((k for k in EMBED_KEYS if k in weight_map), None)
        key or sys.exit(f"no embedding tensor in {repo}: saw {list(weight_map)[:3]} ...")
        shard = hf_hub_download(repo_id=repo, filename=weight_map[key])
        return read_tensor(Path(shard), key), key
    except Exception:                      # single-file checkpoint
        path = Path(hf_hub_download(repo_id=repo, filename="model.safetensors"))
        header, _ = read_header(path)
        key = next((k for k in EMBED_KEYS if k in header), None)
        key or sys.exit(f"no embedding tensor in {repo}: saw {list(header)[:5]} ...")
        return read_tensor(path, key), key


# ---------------------------------------------------------------- self test
def self_test():
    """Write a small safetensors file and read it back, to check the parser."""
    import tempfile

    for dtype, arr in [("F32", np.arange(12, dtype=np.float32).reshape(3, 4)),
                       ("F16", np.arange(12, dtype=np.float16).reshape(3, 4))]:
        blob = arr.tobytes()
        header = json.dumps({"t": {"dtype": dtype, "shape": list(arr.shape),
                                   "data_offsets": [0, len(blob)]}}).encode()
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            f.write(struct.pack("<Q", len(header)) + header + blob)
            path = Path(f.name)
        back = read_tensor(path, "t")
        assert back.shape == arr.shape and np.allclose(back, arr.astype(np.float32)), dtype
        print(f"  {dtype}: shape {back.shape} round-trips")

    # bfloat16, built by truncating float32 to its top 16 bits
    arr = np.array([[1.0, -2.5], [0.25, 1234.0]], dtype=np.float32)
    bf = (arr.view(np.uint32) >> 16).astype("<u2")
    blob = bf.tobytes()
    header = json.dumps({"t": {"dtype": "BF16", "shape": list(arr.shape),
                               "data_offsets": [0, len(blob)]}}).encode()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        f.write(struct.pack("<Q", len(header)) + header + blob)
        path = Path(f.name)
    back = read_tensor(path, "t")
    assert np.allclose(back, arr, rtol=0.01), back
    print(f"  BF16: {back.tolist()} ≈ {arr.tolist()}")
    print("\nthe safetensors reader works; the rest needs a download")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="google/gemma-3-270m",
                    help="any model with a safetensors checkpoint")
    ap.add_argument("--tokenizer", default="", help="defaults to --repo")
    ap.add_argument("--words", type=int, default=50_000)
    ap.add_argument("--out", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok_repo = args.tokenizer or args.repo
    print(f"tokenizer from {tok_repo} ...")
    tok = Tokenizer.from_file(hf_hub_download(repo_id=tok_repo, filename="tokenizer.json"))

    print(f"weights from {args.repo} (this is the big download) ...")
    E, key = find_embeddings(args.repo)
    print(f"  {key}: {E.shape[0]} rows × {E.shape[1]} numbers")

    # keep the rows whose token is a whole word, preferring the word-initial form
    seen = {}
    for i in range(min(E.shape[0], tok.get_vocab_size())):
        piece = tok.id_to_token(i)
        if piece is None:
            continue
        initial = piece.startswith("▁") or piece.startswith("Ġ")
        word = piece[1:] if initial else piece
        if not WORD.match(word) or len(word) < 2:
            continue
        key_ = word.lower()
        if key_ in seen and not initial:      # a word-initial token wins
            continue
        seen[key_] = i
        if len(seen) >= args.words:
            break

    out = Path(args.out) if args.out else RESULTS / (args.repo.split("/")[-1] + "-embeddings.vec")
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        f.write(f"{len(seen)} {E.shape[1]}\n")
        for word, i in seen.items():
            f.write(word + " " + " ".join(f"{x:.6f}" for x in E[i]) + "\n")

    print(f"kept {len(seen)} whole-word rows of {E.shape[0]}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print("\nin Julia:")
    print(f'    llm = load_vectors("{out}"; name = "{args.repo}")')
    print( '    analogy(llm, "man", "king", "woman")')


if __name__ == "__main__":
    sys.exit(main())
