#!/usr/bin/env python3
"""Three follow-ups to the anatomy map (local, small).

A. Causal: transplant one zone's marks from a donor sentence into a base sentence's ring and read.
   who   (7, 9, 29, 35)             base/donor with different subjects -> does the subject change?
   verb  (2, 8, 13, 17, 25, 26)     positive base, negative donor -> does negation appear?
   end   (22, 32, 33)               statement base, question donor -> does it become a question?
   For each: target effect rate, and how much of the base's other words survive. Control: the same
   number of random slots from the same donor.
B. Dictionary: in the 'who' slots, how pure is each symbol with respect to the first word?
C. Stability: does the morning v4 model (Tatoeba-only training) have the same zones?
"""
import collections
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402
import anatomy as AN  # noqa: E402

ZONES = {"who": [7, 9, 29, 35], "verb": [2, 8, 13, 17, 25, 26], "end": [22, 32, 33]}
SUBJ = ["i", "you", "he", "she", "we", "they", "tom", "mary"]


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


def words(s):
    return re.findall(r"[a-z']+|\d+", s.lower())


def rest_kept(base, reading, skip_first=False):
    b, r = words(base), words(reading)
    if skip_first:
        b, r = b[1:], r[1:]
    if not b:
        return 1.0
    cnt = collections.Counter(r)
    return sum(1 for w in b if cnt[w] > 0 and not cnt.subtract([w])) / len(b)


def is_neg(s):
    return any(w in ("not", "never", "no") or w.endswith("n't") for w in words(s))


def main():
    rng = random.Random(0)
    m, _ = H.load(Path("v4/runs/s2/model.pt"), H.device())
    m.eval()
    pool = [s for s in H.load_rows("test") + H.load_rows("mixed_val") if 4 <= len(s.split()) <= 10]
    first = lambda s: words(s)[0] if words(s) else ""  # noqa: E731
    res = {"A": {}}
    # ---------- A
    setups = {
        "who": ([s for s in pool if first(s) in SUBJ and s.endswith(".")], lambda d, b: first(d) != first(b),
                lambda r, d, b: first(r) == first(d), True),
        "verb": ([s for s in pool if not is_neg(s) and s.endswith(".")], None, lambda r, d, b: is_neg(r), False),
        "end": ([s for s in pool if s.endswith(".") and not is_neg(s)], None, lambda r, d, b: r.strip().endswith("?"), False),
    }
    donors = {"who": [s for s in pool if first(s) in SUBJ and s.endswith(".")],
              "verb": [s for s in pool if is_neg(s) and s.endswith(".")],
              "end": [s for s in pool if s.endswith("?")]}
    for z, (bases, ok_pair, effect, skip_first) in setups.items():
        B = rng.sample(bases, 200)
        D = []
        for b in B:
            while True:
                d = rng.choice(donors[z])
                if d != b and (ok_pair is None or ok_pair(d, b)):
                    D.append(d)
                    break
        cb, cd = codes(m, B), codes(m, D)
        zone = ZONES[z]
        x = cb.clone()
        x[:, zone] = cd[:, zone]
        y = cb.clone()
        for i in range(len(y)):
            rs = random.Random(i).sample([s for s in range(H.SLOTS) if s not in zone], len(zone))
            y[i, rs] = cd[i, rs]
        rx, ry, r0 = read(m, x), read(m, y), read(m, cb)
        res["A"][z] = {
            "zone": zone,
            "effect_with_zone": round(float(np.mean([effect(r, d, b) for r, d, b in zip(rx, D, B)])), 3),
            "effect_with_random_slots": round(float(np.mean([effect(r, d, b) for r, d, b in zip(ry, D, B)])), 3),
            "effect_no_transplant": round(float(np.mean([effect(r, d, b) for r, d, b in zip(r0, D, B)])), 3),
            "base_words_kept_zone": round(float(np.mean([rest_kept(b, r, skip_first) for b, r in zip(B, rx)])), 3),
            "base_words_kept_random": round(float(np.mean([rest_kept(b, r, skip_first) for b, r in zip(B, ry)])), 3),
            "examples": [[b, d, r] for b, d, r in list(zip(B, D, rx))[:6]],
        }
    # ---------- B
    texts = rng.sample(pool, min(6000, len(pool)))
    c = codes(m, texts).cpu().numpy()
    fw = [first(t) if first(t) in SUBJ else "other" for t in texts]
    dic = {}
    for s in ZONES["who"]:
        by = collections.defaultdict(collections.Counter)
        for i in range(len(texts)):
            by[int(c[i, s])][fw[i]] += 1
        rows = sorted(by.items(), key=lambda kv: -sum(kv[1].values()))
        purity = sum(cnt.most_common(1)[0][1] for _, cnt in by.items()) / len(texts)
        majority = collections.Counter(fw).most_common(1)[0][1] / len(texts)
        dic[s] = {"purity": round(purity, 3), "majority_baseline": round(majority, 3), "symbols_used": len(by),
                  "top_symbols": [(sym, dict(cnt.most_common(3))) for sym, cnt in rows[:8]]}
    res["B"] = dic
    # ---------- C
    stab = {}
    for name, path in (("v4_morning", "v4/runs/final/model.pt"), ("s2_now", "v4/runs/s2/model.pt")):
        mm, _ = H.load(Path(path), H.device())
        mm.eval()
        cc = codes(mm, texts).cpu().numpy()
        P, bags = zip(*[AN.props(t) for t in texts])
        out = {}
        for f in ("question", "subject_I/we", "subject_he/she/it/they", "negation", "first_word", "last_char", "past"):
            y = [p[f] for p in P]
            hy = AN.entropy(collections.Counter(y))
            ys = y[:]
            random.Random(1).shuffle(ys)
            row = [max(0.0, (AN.mi(cc[:, s].tolist(), y) - AN.mi(cc[:, s].tolist(), ys)) / hy) for s in range(H.SLOTS)]
            out[f] = {"best_slots": sorted(range(H.SLOTS), key=lambda s: -row[s])[:4], "best_share": round(max(row), 3)}
        stab[name] = out
    res["C"] = stab
    Path("v4/runs/s2/zones_causal.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
