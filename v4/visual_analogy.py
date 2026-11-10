#!/usr/bin/env python3
"""Analogies done on the DRAWINGS only (no arithmetic inside T5).

1. pixel arithmetic: image(b) - image(a) + image(c), clipped to [0,1], read by the eyes + reader.
   Control: image(c) alone; and the same arithmetic with (a, b) from another relation.
2. graphical rule: from the rings of training pairs of a relation, find slots whose mark changes
   consistently (consensus: slot changes in >= 40% of pairs; new mark = most frequent new mark,
   or "keep the pair's own mark" is not available, so the rule sets that mark). Apply the rule to
   the rings of held-out words (leave-one-out over pairs) and read. Control: same number of
   random slot edits. All words inside "The answer was {w}."
"""
import collections
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402
import analogy_probe as A  # noqa: E402


def has(r, w):
    return bool(re.search(rf"\b{re.escape(w.lower())}\b", r.lower()))


@torch.no_grad()
def main():
    m, _ = H.load(Path("v4/runs/s2/model.pt"), H.device())
    m.eval()
    fams = A.FAMILIES
    words = sorted({w for f in fams.values() for p in f for w in p})
    sym = dict(zip(words, A.quantize(m, A.raw(m, words))))
    img = {w: m.draw(m.fsq_to_mark[sym[w]][None])[0] for w in words}
    rng = random.Random(0)
    K = m.books.shape[1]
    res, ex = {}, []
    for fam, pairs in fams.items():
        # 1. pixel arithmetic
        hits, base, other = [], [], []
        for (a, b), (c, d) in [(p, q) for p in pairs for q in pairs if p != q]:
            z = (img[b] - img[a] + img[c]).clamp(0, 1)
            a2, b2 = rng.choice(fams[rng.choice([f for f in fams if f != fam])])
            z2 = (img[b2] - img[a2] + img[c]).clamp(0, 1)
            r1, r0, r2 = m.generate(m.eye_probs(torch.stack([z, img[c], z2])))
            hits.append(has(r1, d)); base.append(has(r0, d)); other.append(has(r2, d))
            if len(ex) < 40 and fam in ("male->female", "singular->plural"):
                ex.append(f"pixels: img({b}) - img({a}) + img({c}) -> '{r1}' (expected {d})")
        # 2. graphical rule, leave-one-out
        rule_hits, rand_hits, sizes = [], [], []
        for i, (c, d) in enumerate(pairs):
            train = [p for j, p in enumerate(pairs) if j != i]
            per = collections.defaultdict(collections.Counter)
            for a, b in train:
                for s in range(H.SLOTS):
                    if sym[a][s] != sym[b][s]:
                        per[s][int(sym[b][s])] += 1
            rule = {s: cnt.most_common(1)[0][0] for s, cnt in per.items() if sum(cnt.values()) >= 0.4 * len(train)}
            sizes.append(len(rule))
            x = sym[c].clone()
            for s, v in rule.items():
                x[s] = v
            y = sym[c].clone()
            for s in random.Random(i).sample(range(H.SLOTS), max(1, len(rule))):
                y[s] = (y[s] + random.Random(100 + s).randrange(1, K)) % K
            r1, r2 = m.generate(m.eye_probs(m.draw(m.fsq_to_mark[torch.stack([x, y])])))
            rule_hits.append(has(r1, d)); rand_hits.append(has(r2, d))
            ex.append(f"rule[{fam}] on ring({c}) -> '{r1}' (expected {d})")
        res[fam] = {"pixel_arithmetic": round(float(np.mean(hits)), 3), "pixel_no_change": round(float(np.mean(base)), 3),
                    "pixel_other_relation": round(float(np.mean(other)), 3),
                    "graphic_rule": round(float(np.mean(rule_hits)), 3), "rule_size_slots_mean": round(float(np.mean(sizes)), 1),
                    "random_edit_same_size": round(float(np.mean(rand_hits)), 3)}
    Path("v4/runs/s2/visual_analogy.json").write_text(json.dumps({"results": res, "examples": ex}, indent=1))
    print(json.dumps(res, indent=1))
    print("\n".join(ex))


if __name__ == "__main__":
    main()
