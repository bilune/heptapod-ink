#!/usr/bin/env python3
"""Where is the content? Controlled sentences "{Subj} {verb} the {noun}." (6 x 12 x 40 = 2,880).

1. Map: mutual information between each slot's symbol and subject / verb / noun identity / noun
   category (5 categories x 8 nouns), normalised by the property's entropy, shuffle-corrected.
2. Category vs identity: within the best noun slots, how much of the noun's category is known vs
   the exact noun (do symbols group nouns by meaning?).
3. Causal: transplant the top-k noun slots from a donor (same subject and verb, different noun)
   into a base ring; read. Success = the reading has the donor's noun and keeps subject + verb.
   k = 1..12, vs the same number of random non-zone slots.
"""
import collections
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402

SUBJ = ["I", "She", "Tom", "We", "They", "My father"]
VERBS = ["saw", "liked", "found", "bought", "painted", "washed", "moved", "wanted", "needed", "sold", "watched", "drew"]
CATS = {"animal": ["dog", "cat", "horse", "bird", "cow", "fish", "rabbit", "sheep"],
        "food": ["apple", "bread", "cheese", "soup", "cake", "rice", "egg", "banana"],
        "vehicle": ["car", "bus", "train", "boat", "bike", "truck", "plane", "taxi"],
        "place": ["school", "church", "hotel", "park", "beach", "office", "library", "market"],
        "tool": ["hammer", "knife", "ladder", "shovel", "saw", "needle", "brush", "rope"]}
NOUN_CAT = {n: c for c, ns in CATS.items() for n in ns}
ZONE_SLOTS = {7, 9, 29, 35, 2, 8, 13, 17, 25, 26, 22, 32, 33, 15, 19, 28}


def H_(c):
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values() if v)


def mi(x, y):
    return H_(collections.Counter(x)) + H_(collections.Counter(y)) - H_(collections.Counter(zip(x, y)))


def norm_mi(x, y, seed=1):
    ys = list(y)
    random.Random(seed).shuffle(ys)
    return max(0.0, (mi(x, y) - mi(x, ys)) / H_(collections.Counter(y)))


@torch.no_grad()
def codes(m, texts):
    out = []
    for i in range(0, len(texts), 128):
        raw = m.writer_out(texts[i:i + 128]).float()
        out.append(((raw[:, :, None, :] - m.books[None]) ** 2).sum(-1).argmin(-1))
    return torch.cat(out)


@torch.no_grad()
def read(m, sym):
    preds = []
    for i in range(0, len(sym), 64):
        preds += m.generate(F.one_hot(m.fsq_to_mark[sym[i:i + 64]], m.script.n_marks).float())
    return preds


def main():
    m, _ = H.load(Path("v4/runs/s2/model.pt"), H.device())
    m.eval()
    items = [(s, v, n) for s in SUBJ for v in VERBS for n in NOUN_CAT]
    texts = [f"{s} {v} the {n}." for s, v, n in items]
    C = codes(m, texts)
    c = C.cpu().numpy()
    props = {"subject": [s for s, _, _ in items], "verb": [v for _, v, _ in items],
             "noun": [n for _, _, n in items], "noun_category": [NOUN_CAT[n] for _, _, n in items]}
    table = {p: [norm_mi(c[:, k].tolist(), y) for k in range(H.SLOTS)] for p, y in props.items()}
    res = {"map": {p: {"best_slots": sorted(range(H.SLOTS), key=lambda k: -r[k])[:6], "best_share": round(max(r), 3),
                       "sum_share": round(sum(r), 2)} for p, r in table.items()}}
    # how many slots does it take to pin down the noun? (joint purity of the top-k noun slots)
    order = sorted(range(H.SLOTS), key=lambda k: -table["noun"][k])
    pin = {}
    for k in (1, 2, 3, 4, 6, 8):
        keys = [tuple(c[i, order[:k]]) for i in range(len(items))]
        by = collections.defaultdict(collections.Counter)
        for key, n in zip(keys, props["noun"]):
            by[key][n] += 1
        pin[k] = round(sum(cnt.most_common(1)[0][1] for cnt in by.values()) / len(items), 3)
    res["noun_identified_from_top_k_noun_slots"] = pin
    # category vs identity inside the best noun slots
    best = order[:4]
    res["category_vs_identity_in_best_noun_slots"] = {k: {"noun": round(table["noun"][k], 3), "category": round(table["noun_category"][k], 3)} for k in best}
    # do symbols in the best noun slot group nouns of the same category?
    k0 = best[0]
    groups = collections.defaultdict(set)
    for i, (_, _, n) in enumerate(items):
        groups[int(c[i, k0])].add(n)
    res["symbol_groups_best_noun_slot"] = {str(sym): sorted(ns) for sym, ns in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:12]}
    # causal transplant of top-k noun slots
    rng = random.Random(0)
    idx = {t: i for i, t in enumerate(texts)}
    trials = []
    for _ in range(300):
        s, v = rng.choice(SUBJ), rng.choice(VERBS)
        n1, n2 = rng.sample(list(NOUN_CAT), 2)
        trials.append((idx[f"{s} {v} the {n1}."], idx[f"{s} {v} the {n2}."], s, v, n2))
    nonzone = [k for k in range(H.SLOTS) if k not in ZONE_SLOTS]
    curve = {}
    for k in (1, 2, 3, 4, 6, 8, 12):
        top = order[:k]
        x = C[[b for b, _, _, _, _ in trials]].clone()
        y = x.clone()
        for row, (b, d, *_r) in enumerate(trials):
            x[row, top] = C[d, top]
            rs = random.Random(row + k).sample(nonzone, k)
            y[row, rs] = C[d, rs]
        rx, ry = read(m, x), read(m, y)

        def ok(r, s, v, n2):
            w = r.lower()
            return bool(re.search(rf"\b{n2}\b", w)) and w.startswith(s.lower()) and bool(re.search(rf"\b{v}\b", w))
        curve[k] = {"slots": top, "noun_swapped_rest_kept": round(float(np.mean([ok(r, s, v, n2) for r, (_, _, s, v, n2) in zip(rx, trials)])), 3),
                    "random_nonzone_slots": round(float(np.mean([ok(r, s, v, n2) for r, (_, _, s, v, n2) in zip(ry, trials)])), 3),
                    "example": rx[0], "example_target": f"{trials[0][2]} {trials[0][3]} the {trials[0][4]}."}
    res["transplant_noun_slots"] = curve
    res["plain_reading_exact"] = round(float(np.mean([H.norm(r) == H.norm(t) for r, t in zip(read(m, C), texts)])), 3)
    Path("v4/runs/s2/content_probe.json").write_text(json.dumps({**res, "table": table}, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
