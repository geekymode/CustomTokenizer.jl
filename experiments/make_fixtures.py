"""Record exactly what the reference tokenizers do, as a test oracle for the
Julia implementation.

    python experiments/make_fixtures.py

Writes test/fixtures/tokenizers.json: for each tokenizer, the local path of its
tokenizer.json and, for a list of texts, the ids and token strings the Rust
implementation produces. The Julia loader must reproduce these exactly.
"""

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "test" / "fixtures"

REPOS = {
    "gpt2": "gpt2",                          # = r50k_base, GPT-3's tokenizer
    "gemma4": "google/gemma-4-E2B",
    "qwen25": "Qwen/Qwen2.5-0.5B",
}

TEXTS = [
    "the cat drinks milk",
    "The cat drinks milk.",
    "man king woman queen",
    "In 2026 the price rose 7900%.",
    "def f(x):\n    return x + 1\n",
    "a     b",
    "naïve café 🐈",
    "antidisestablishmentarianism",
    "  leading and trailing  ",
    "",
]


def main():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    out = {}
    for name, repo in REPOS.items():
        path = hf_hub_download(repo_id=repo, filename="tokenizer.json")
        tok = Tokenizer.from_file(path)
        cases = []
        for text in TEXTS:
            enc = tok.encode(text, add_special_tokens=False)
            cases.append({"text": text, "ids": enc.ids, "tokens": enc.tokens,
                          "decoded": tok.decode(enc.ids)})
        out[name] = {"repo": repo, "path": path, "vocab_size": tok.get_vocab_size(),
                     "cases": cases}
        print(f"{name:8} {repo:24} {tok.get_vocab_size():>7} tokens, {len(cases)} cases")

    FIXTURES.mkdir(parents=True, exist_ok=True)
    dest = FIXTURES / "tokenizers.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nwrote {dest} ({dest.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
