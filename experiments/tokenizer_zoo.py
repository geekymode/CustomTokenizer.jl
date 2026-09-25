"""Compare real LLM tokenizers with the naive one in CustomTokenizer.jl.

Downloads only the tokenizer files (a few MB each) — no model weights are
needed to study tokenization.

    pip install tiktoken tokenizers huggingface_hub
    python experiments/tokenizer_zoo.py

Writes experiments/results/tokenizers.json for the Julia plotting script and
prints the tables below.
"""

import json
import os
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


# --------------------------------------------------------------- tokenizers
class Tok:
    """One tokenizer behind a common interface."""

    def __init__(self, name, family, encode, decode, vocab_size, note=""):
        self.name, self.family = name, family
        self._encode, self._decode = encode, decode
        self.vocab_size, self.note = vocab_size, note

    def ids(self, text):
        return self._encode(text)

    def pieces(self, text):
        """Token strings, via round-tripping each id on its own where possible."""
        out = []
        for i in self.ids(text):
            try:
                out.append(self._decode([i]))
            except Exception:
                out.append(f"<{i}>")
        return out

    def decode(self, ids):
        return self._decode(ids)


def load_tokenizers():
    import tiktoken
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer as HFTokenizer

    toks = []

    for name, enc_name, note in [
        ("GPT-3", "r50k_base", "davinci; also GPT-2"),
        ("GPT-3.5/4", "cl100k_base", ""),
        ("GPT-4o", "o200k_base", ""),
    ]:
        enc = tiktoken.get_encoding(enc_name)
        toks.append(Tok(name, "tiktoken BPE (bytes)", enc.encode, enc.decode,
                        enc.n_vocab, note))

    for name, repo, note in [
        ("Gemma 4", "google/gemma-4-E2B", "SentencePiece, 262k"),
        ("Gemma 2", "philschmid/gemma-tokenizer-chatml", "SentencePiece, 256k"),
        ("Qwen 2.5", "Qwen/Qwen2.5-0.5B", "BPE, multilingual"),
    ]:
        path = hf_hub_download(repo_id=repo, filename="tokenizer.json")
        hf = HFTokenizer.from_file(path)
        toks.append(Tok(name, "HF tokenizer.json",
                        lambda t, hf=hf: hf.encode(t).ids,
                        lambda ids, hf=hf: hf.decode(ids),
                        hf.get_vocab_size(), note))

    # the tokenizer from the Julia package, for contrast
    word_re = re.compile(r"[A-Za-z]+")
    toks.append(Tok("CustomTokenizer.jl", "whitespace/word",
                    lambda t: [hash(w) for w in word_re.findall(t.lower())],
                    lambda ids: "<not reversible>",
                    None, "lowercase, letters only"))
    return toks


# --------------------------------------------------------------- test material
TEXTS = {
    "english": "The cat drinks milk while the queen wears a crown in her castle.",
    "german": "Die Katze trinkt Milch, während die Königin eine Krone trägt.",
    "hindi": "बिल्ली दूध पीती है जबकि रानी अपने महल में मुकुट पहनती है।",
    "chinese": "猫在喝牛奶，女王在她的城堡里戴着王冠。",
    "code": 'def total(xs):\n    """Sum a list."""\n    s = 0\n    for x in xs:\n        s += x\n    return s\n',
    "numbers": "In 2026 the price rose from 1234.56 to 98765.43, a 7900% change.",
    "emoji": "the cat 🐈 drinks milk 🥛 — naïve café façade 🫩",
}

PROBES = {
    "a year": "2026",
    "a long number": "1234567",
    "indentation": "def f():\n    return 1",
    "repeated spaces": "a     b",
    "an emoji": "🫩",
    "capitalisation": "Cat cat CAT",
    "a rare word": "antidisestablishmentarianism",
}


# --------------------------------------------------------------- measurements
def table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def main():
    toks = load_tokenizers()
    results = {"vocab": {}, "fertility": {}, "probes": {}, "roundtrip": {}}

    print("\n=== the tokenizers ===\n")
    table([[t.name, t.family, t.vocab_size or "—", t.note] for t in toks],
          ["tokenizer", "kind", "vocabulary", "note"])
    for t in toks:
        results["vocab"][t.name] = t.vocab_size

    print("\n=== tokens per text (fewer is better: less to pay for, more context) ===\n")
    rows = []
    for key, text in TEXTS.items():
        words = len(text.split())
        row = [key, len(text.encode()), words]
        for t in toks:
            n = len(t.ids(text))
            row.append(n)
            results["fertility"].setdefault(key, {})[t.name] = {
                "tokens": n, "bytes": len(text.encode()), "words": words}
        rows.append(row)
    table(rows, ["text", "bytes", "words"] + [t.name for t in toks])

    print("\n=== bytes per token (higher = more text in the same context window) ===\n")
    rows = []
    for key, text in TEXTS.items():
        b = len(text.encode())
        rows.append([key] + [f"{b / max(1, len(t.ids(text))):.2f}" for t in toks])
    table(rows, ["text"] + [t.name for t in toks])

    print("\n=== how each tokenizer cuts things up ===\n")
    for label, probe in PROBES.items():
        print(f"{label}:  {probe!r}")
        for t in toks:
            if t.name == "CustomTokenizer.jl":
                shown = re.findall(r"[A-Za-z]+", probe.lower())
            else:
                shown = t.pieces(probe)
            results["probes"].setdefault(label, {})[t.name] = [str(s) for s in shown]
            print(f"    {t.name:20} {len(shown):>3}  {shown}")
        print()

    print("=== decode(encode(x)) == x ? ===\n")
    rows = []
    for t in toks:
        ok = []
        for key, text in TEXTS.items():
            try:
                ok.append(t.decode(t.ids(text)) == text)
            except Exception:
                ok.append(False)
        results["roundtrip"][t.name] = {k: v for k, v in zip(TEXTS, ok)}
        rows.append([t.name, f"{sum(ok)}/{len(ok)}",
                     ", ".join(k for k, v in zip(TEXTS, ok) if not v) or "—"])
    table(rows, ["tokenizer", "exact", "fails on"])

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "tokenizers.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # plain TSVs so the Julia plotting script needs no JSON dependency
    fert = RESULTS / "fertility.tsv"
    with fert.open("w") as f:
        f.write("text\tbytes\twords\t" + "\t".join(t.name for t in toks) + "\n")
        for key, text in TEXTS.items():
            f.write(f"{key}\t{len(text.encode())}\t{len(text.split())}\t" +
                    "\t".join(str(len(t.ids(text))) for t in toks) + "\n")
    voc = RESULTS / "vocab.tsv"
    with voc.open("w") as f:
        f.write("tokenizer\tvocab\n")
        for t in toks:
            t.vocab_size and f.write(f"{t.name}\t{t.vocab_size}\n")
    print(f"\nwrote {out}, {fert}, {voc}")


if __name__ == "__main__":
    sys.exit(main())
