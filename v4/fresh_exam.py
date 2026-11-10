#!/usr/bin/env python3
"""Build the fresh held-out exam for the sender-search work (v4 test stays retired).

fresh_natural: short sentences (3-12 words) from UD English EWT dev+test
  (CC BY-SA 4.0, web English, a corpus never used in training), T5 round-trip
  clean, deduplicated against every Tatoeba sentence we have.
fresh_random: 1,000 new random word combinations (new seed), excluding all
  existing evaluation strings.
Written next to the Tatoeba splits so load_rows() and the Modal image see them.
"""
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v4"))
import heptapod_t5 as H  # noqa: E402

tok = AutoTokenizer.from_pretrained(H.T5)
known = set()
for line in (ROOT / "research" / "tatoeba" / "eng_sentences.tsv").read_text(encoding="utf-8").splitlines():
    parts = line.split("\t")
    if len(parts) >= 3:
        known.add(" ".join(parts[2].split()).lower())
cands = []
for f in sorted((ROOT / "research" / "ud_ewt").glob("*.conllu")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("# text = "):
            s = " ".join(line[len("# text = "):].split())
            n = len(s.split())
            if 3 <= n <= 12 and s.lower() not in known and any(c.isalpha() for c in s):
                ids = tok(s).input_ids
                if len(ids) <= 38 and tok.decode(ids, skip_special_tokens=True) == s:
                    cands.append(s)
cands = list(dict.fromkeys(cands))
random.Random(2026).shuffle(cands)
natural = cands[:1000]
s = H.Sampler(random_frac=1.0, seed=2026)
used = set(natural) | {x.lower() for x in natural}
rand = []
while len(rand) < 1000:
    x = s.random_sentence()
    if x not in used:
        used.add(x)
        rand.append(x)
for name, rows, cat in (("fresh_natural", natural, "ud_ewt"), ("fresh_random", rand, "random")):
    (H.DATA / f"{name}.json").write_text(json.dumps([{"sentence": r, "category": cat} for r in rows]))
print({"natural_candidates": len(cands), "fresh_natural": len(natural), "fresh_random": len(rand)})
print(natural[:6], rand[:3])
