"""Download published word vectors and write them as plain text.

    python experiments/fetch_pretrained.py [--words 50000] [--dim 50]

Fetches GloVe (Wikipedia + Gigaword, 6B tokens) from the Hugging Face mirror
and writes experiments/results/glove-<dim>d-<words>k.vec in the usual
"word v1 v2 ..." format, which CustomTokenizer.load_vectors reads.

Only the first `--words` entries are kept; the file is sorted by frequency, so
that keeps the words that matter and skips a very long tail.
"""

import argparse
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=50, choices=[50, 100, 200, 300])
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from gensim.models import KeyedVectors

    repo = f"fse/glove-wiki-gigaword-{args.dim}"
    print(f"downloading {repo} ...")
    model_file = hf_hub_download(repo_id=repo, filename=f"glove-wiki-gigaword-{args.dim}.model")
    hf_hub_download(repo_id=repo, filename=f"glove-wiki-gigaword-{args.dim}.model.vectors.npy")
    kv = KeyedVectors.load(model_file)

    n = min(args.words, len(kv.index_to_key))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"glove-{args.dim}d-{n // 1000}k.vec"
    with out.open("w") as f:
        f.write(f"{n} {kv.vector_size}\n")
        for w in kv.index_to_key[:n]:
            f.write(w + " " + " ".join(f"{x:.6f}" for x in kv[w]) + "\n")

    print(f"{len(kv.index_to_key)} words available, wrote the first {n}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print("\nin Julia:")
    print(f'    glove = load_vectors("{out}"; name = "GloVe {args.dim}d")')


if __name__ == "__main__":
    sys.exit(main())
