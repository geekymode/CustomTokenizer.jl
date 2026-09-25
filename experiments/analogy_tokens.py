"""Do the analogy words survive as single tokens?

A tokenizer has no vectors, so "man - woman + queen" cannot be computed *in* a
tokenizer. What a tokenizer decides is whether a word is one unit or several,
and that decides whether the model behind it has a single vector for that word
at all. A word split into three pieces has three vectors, none of which is "the
vector for the word".

    python experiments/analogy_tokens.py

Prints, for every tokenizer, how each analogy word is cut up, and flags the
questions where some word is not a single token.
"""

import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

QUESTIONS = [
    ("man", "king", "woman", "queen"),
    ("man", "uncle", "woman", "aunt"),
    ("boy", "son", "girl", "daughter"),
    ("france", "paris", "italy", "rome"),
    ("germany", "berlin", "japan", "tokyo"),
    ("good", "better", "bad", "worse"),
    ("walk", "walking", "swim", "swimming"),
]


def load_tokenizers():
    """Same set as tokenizer_zoo.py, but tolerant of a repo being unreachable."""
    import tiktoken
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer as HFTokenizer

    toks = []
    for name, enc in [("GPT-3", "r50k_base"), ("GPT-3.5/4", "cl100k_base"),
                      ("GPT-4o", "o200k_base")]:
        e = tiktoken.get_encoding(enc)
        toks.append((name, lambda t, e=e: [e.decode([i]) for i in e.encode(t)]))

    for name, repo in [("Gemma 4", "google/gemma-4-E2B"),
                       ("Gemma 2", "philschmid/gemma-tokenizer-chatml"),
                       ("Qwen 2.5", "Qwen/Qwen2.5-0.5B"),
                       ("GPT-2", "gpt2")]:
        try:
            hf = HFTokenizer.from_file(hf_hub_download(repo_id=repo, filename="tokenizer.json"))
        except Exception as exc:                       # offline, rate limited, gated
            print(f"  (skipping {name}: {type(exc).__name__})", file=sys.stderr)
            continue
        toks.append((name, lambda t, hf=hf: [p.replace("▁", " ")
                                             for p in hf.encode(t, add_special_tokens=False).tokens]))
    return toks


def main():
    toks = load_tokenizers()
    words = sorted({w for q in QUESTIONS for w in q})

    print("\n=== how each analogy word is tokenized (with its leading space) ===\n")
    head = f"{'word':<10}" + "".join(f"{n:<22}" for n, _ in toks)
    print(head)
    print("-" * len(head))
    single = {n: 0 for n, _ in toks}
    for w in words:
        row = f"{w:<10}"
        for name, split in toks:
            pieces = split(" " + w)
            len(pieces) == 1 and (single.__setitem__(name, single[name] + 1))
            shown = "|".join(p.strip() or "␣" for p in pieces)
            row += f"{shown:<22}" if len(shown) < 21 else f"{shown[:19]}… "
        print(row)

    print("\n=== single-token words ===\n")
    for name, _ in toks:
        print(f"  {name:<12} {single[name]:>2}/{len(words)} words are one token")

    print("\n=== questions where every word is a single token ===\n")
    for a, b, c, want in QUESTIONS:
        marks = []
        for name, split in toks:
            ok = all(len(split(" " + w)) == 1 for w in (a, b, c, want))
            marks.append(f"{name}: {'yes' if ok else 'no '}")
        print(f"  {a}/{b}/{c}/{want:<9} " + "  ".join(marks))

    print("""
Only for those questions does the model behind the tokenizer have one input
vector per word. Everywhere else the word is a sequence, and any analogy over
it has to pool several vectors — which is a different operation, with a
different failure mode (see the sub-word experiment).""")


if __name__ == "__main__":
    sys.exit(main())
