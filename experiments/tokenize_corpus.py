"""Tokenize one corpus two ways, so the same text can be trained on twice.

    python experiments/tokenize_corpus.py path/to/corpus.txt [--limit 200000]

Writes into experiments/results/:

  stream_word.txt    one sentence per line, whitespace/word tokens (this
                     package's rule: lowercase, letters only)
  stream_gemma.txt   the same sentences as Gemma 4 sub-word pieces
  word_pieces.tsv    word -> the pieces it is made of, so the Julia side can
                     build a word vector by pooling its pieces
  tokenize_stats.tsv counts for both streams

Only the tokenizer file is downloaded (a few MB); no model weights are needed.
"""

import argparse
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SENTENCE = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[A-Za-z]+")
GEMMA_REPO = "google/gemma-4-E2B"


def sentences_of(path, limit):
    text = Path(path).read_text(errors="replace")
    if limit:
        text = text[:limit]
    out = []
    for block in text.splitlines():
        for s in SENTENCE.split(block):
            s = s.strip()
            len(s) > 1 and out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=0, help="characters to read (0 = all)")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(repo_id=GEMMA_REPO, filename="tokenizer.json"))
    sents = sentences_of(args.corpus, args.limit)
    RESULTS.mkdir(exist_ok=True)

    word_lines, gemma_lines = [], []
    vocab_words, vocab_pieces = set(), set()
    n_word_tokens = n_gemma_tokens = 0

    for s in sents:
        words = [w.lower() for w in WORD.findall(s)]
        if len(words) < 2:
            continue
        word_lines.append(" ".join(words))
        vocab_words.update(words)
        n_word_tokens += len(words)

        # Sub-word pieces, without the automatic <bos> on every sentence.
        # The text is lowercased first so that the ONLY difference between the
        # two streams is word vs sub-word units, not capitalisation.
        pieces = tok.encode(s.lower(), add_special_tokens=False).tokens
        pieces = [p.replace(" ", "▁") for p in pieces if p.strip() != ""]
        gemma_lines.append(" ".join(pieces))
        vocab_pieces.update(pieces)
        n_gemma_tokens += len(pieces)

    (RESULTS / "stream_word.txt").write_text("\n".join(word_lines) + "\n")
    (RESULTS / "stream_gemma.txt").write_text("\n".join(gemma_lines) + "\n")

    # how each word breaks into pieces (word-initial form, i.e. preceded by a space)
    with (RESULTS / "word_pieces.tsv").open("w") as f:
        for w in sorted(vocab_words):
            ps = tok.encode(" " + w, add_special_tokens=False).tokens
            ps = [p.replace(" ", "▁") for p in ps if p.strip() != ""]
            f.write(w + "\t" + " ".join(ps) + "\n")

    whole = sum(1 for w in vocab_words
                if len(tok.encode(" " + w, add_special_tokens=False).tokens) == 1)
    with (RESULTS / "tokenize_stats.tsv").open("w") as f:
        f.write("measure\tword\tgemma\n")
        f.write(f"sentences\t{len(word_lines)}\t{len(gemma_lines)}\n")
        f.write(f"tokens\t{n_word_tokens}\t{n_gemma_tokens}\n")
        f.write(f"distinct\t{len(vocab_words)}\t{len(vocab_pieces)}\n")

    print(f"sentences            {len(word_lines)}")
    print(f"tokens   word-level  {n_word_tokens}")
    print(f"         sub-word    {n_gemma_tokens}  ({n_gemma_tokens / n_word_tokens:.2f}× as many)")
    print(f"distinct word-level  {len(vocab_words)}")
    print(f"         sub-word    {len(vocab_pieces)} pieces")
    print(f"whole words kept as one piece: {whole}/{len(vocab_words)} "
          f"({100 * whole / len(vocab_words):.0f}%)")
    print(f"\nwrote four files to {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
