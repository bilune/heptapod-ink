#!/usr/bin/env python3
"""Anatomy of the logogram: which slots carry which properties of the sentence?

Encode N held-out natural sentences (mixed_val + Tatoeba dev/test, never trained on) with the plain
writer. For each slot (36) and each sentence property, compute the mutual information between the
slot's symbol (81 values) and the property, normalised by the property's entropy (share of the
property's uncertainty that one slot alone resolves). A shuffled baseline (property permuted) is
subtracted to remove the small-sample bias.

Properties (simple surface heuristics, no parser): question, negation, first-person/second/third
subject, plural-ish subject, past (was/were/did/-ed after the subject), future (will/'ll/going to),
contains a number, contains a proper name (capitalised word not first), length bin, first word,
last word, and presence of each of the 40 most frequent content words.
"""
import collections
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402

STOP = set("the a an to of and in is it that i you he she we they was for on with be this my your have not do at are me his her as but so what can will".split())


def props(s):
    w = re.findall(r"[A-Za-z']+|\d+", s)
    lw = [x.lower() for x in w]
    first = lw[0] if lw else ""
    p = {
        "question": s.strip().endswith("?"),
        "negation": any(x in ("not", "never", "no") or x.endswith("n't") for x in lw),
        "subject_I/we": first in ("i", "we"),
        "subject_you": first == "you",
        "subject_he/she/it/they": first in ("he", "she", "it", "they"),
        "past": bool(re.search(r"\b(was|were|did|had)\b", s.lower())) or any(x.endswith("ed") for x in lw[1:3]),
        "future": bool(re.search(r"\b(will|going to)\b|'ll\b", s.lower())),
        "has_number": bool(re.search(r"\d", s)),
        "has_name": any(x[:1].isupper() and x.lower() not in ("i",) for x in w[1:]),
        "length": "short" if len(w) <= 5 else "medium" if len(w) <= 8 else "long",
        "first_word": first if first in ("i", "you", "he", "she", "it", "we", "they", "the", "this", "what", "tom", "there", "my", "do", "a") else "other",
        "last_char": s.strip()[-1:] if s.strip()[-1:] in ".?!" else "none",
    }
    return p, set(lw)


def entropy(c):
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values() if v)


def mi(x, y):
    joint = collections.Counter(zip(x, y))
    return entropy(collections.Counter(x)) + entropy(collections.Counter(y)) - entropy(joint)


@torch.no_grad()
def main():
    rng = random.Random(0)
    m, _ = H.load(Path("v4/runs/s2/model.pt"), H.device())
    m.eval()
    texts = H.load_rows("mixed_val") + H.load_rows("test")
    texts = rng.sample(texts, min(8000, len(texts)))
    codes = []
    for i in range(0, len(texts), 128):
        raw = m.writer_out(texts[i:i + 128]).float()
        codes.append(((raw[:, :, None, :] - m.books[None]) ** 2).sum(-1).argmin(-1).cpu())
    codes = torch.cat(codes).numpy()
    P, bags = zip(*[props(t) for t in texts])
    names = list(P[0])
    vocab = collections.Counter(w for b in bags for w in b if w not in STOP and len(w) > 2)
    words = [w for w, _ in vocab.most_common(40)]
    feats = {n: [p[n] for p in P] for n in names}
    feats.update({f"word:{w}": [w in b for b in bags] for w in words})
    table = {}
    for f, y in feats.items():
        hy = entropy(collections.Counter(y))
        if hy < 0.05:
            continue
        ys = y[:]
        random.Random(1).shuffle(ys)
        row = []
        for s in range(H.SLOTS):
            x = codes[:, s].tolist()
            row.append(max(0.0, (mi(x, y) - mi(x, ys)) / hy))
        table[f] = row
    out = {"n_sentences": len(texts), "features": {f: {"best_slots": sorted(range(H.SLOTS), key=lambda s: -r[s])[:4],
                                                      "best_share": round(max(r), 3), "sum_share": round(sum(r), 3)}
                                                  for f, r in table.items()}}
    Path("v4/runs/s2/anatomy.json").write_text(json.dumps({**out, "table": table}, indent=1))
    for f, v in sorted(out["features"].items(), key=lambda kv: -kv[1]["best_share"]):
        print(f"{f:28s} best slots {v['best_slots']}  best single-slot share {v['best_share']:.3f}  sum over slots {v['sum_share']:.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = names + [f"word:{w}" for w in words[:20]]
    keys = [k for k in keys if k in table]
    M = np.array([table[k] for k in keys])
    plt.figure(figsize=(13, 0.32 * len(keys) + 2))
    plt.imshow(M, aspect="auto", cmap="Blues")
    plt.yticks(range(len(keys)), keys, fontsize=8)
    plt.xticks(range(H.SLOTS), range(H.SLOTS), fontsize=7)
    plt.xlabel("ring slot")
    plt.colorbar(label="share of the property resolved by that slot alone")
    plt.title("Logogram anatomy: which ring slot carries which sentence property", fontsize=11)
    plt.tight_layout()
    plt.savefig("v4/runs/s2/anatomy.png", dpi=100)
    print("v4/runs/s2/anatomy.png")


if __name__ == "__main__":
    main()
