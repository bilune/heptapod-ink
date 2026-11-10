#!/usr/bin/env python3
"""Locked final exam (debate 4 protocol). Built once, fingerprinted, evaluated once at the end.

locked_natural: 3-12-word sentences from UD English GUM, PUD, LinES, ParTUT (dev+test),
  T5 round-trip clean, deduplicated against Tatoeba, EWT and each other; up to 2,000,
  sampled per source document id is not available for all treebanks, so sentence-level.
locked_random: 2,000 weighted combinations (3-10 words, the training sampler's
  distribution) with a new seed, plus locked_uniform10: 1,000 uniform 10-word combinations.
"""
import hashlib
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
    p = line.split("\t")
    if len(p) >= 3:
        known.add(" ".join(p[2].split()).lower())
for s in H.load_rows("fresh_natural"):
    known.add(s.lower())
cands = []
for f in sorted((ROOT / "research" / "ud_locked").glob("*.conllu")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("# text = "):
            s = " ".join(line[9:].split())
            if 3 <= len(s.split()) <= 12 and s.lower() not in known and any(c.isalpha() for c in s):
                ids = tok(s).input_ids
                if len(ids) <= 38 and tok.decode(ids, skip_special_tokens=True) == s:
                    cands.append((f.stem, s))
                    known.add(s.lower())
random.Random(4242).shuffle(cands)
natural = cands[:2000]
vocab = json.loads((H.DATA / "random_vocab.json").read_text())
held = {x.lower() for x in H.Sampler(random_frac=1.0, seed=0).held}
rng = random.Random(777)
rand, uni = [], []
while len(rand) < 2000:
    x = " ".join(rng.choices(vocab["words"], weights=vocab["weights"], k=rng.randint(3, 10)))
    if x not in held and x not in rand:
        rand.append(x)
while len(uni) < 1000:
    uni.append(" ".join(rng.choice(vocab["words"]) for _ in range(10)))
out = {"locked_natural": [{"sentence": s, "category": src} for src, s in natural],
       "locked_random": [{"sentence": s, "category": "random"} for s in rand],
       "locked_uniform10": [{"sentence": s, "category": "uniform10"} for s in uni]}
fp = {}
for name, rows in out.items():
    data = json.dumps(rows)
    (H.DATA / f"{name}.json").write_text(data)
    fp[name] = {"n": len(rows), "sha256": hashlib.sha256(data.encode()).hexdigest()}
(ROOT / "v4" / "LOCKED_EXAM.json").write_text(json.dumps(fp, indent=1))
print(json.dumps(fp, indent=1), "candidates", len(cands))
