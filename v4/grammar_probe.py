#!/usr/bin/env python3
"""The grammar of the ring: do grammatical operations leave consistent, transplantable mark edits?

Five operations, 60 fresh minimal pairs each (base -> transformed), built from small word lists:
past tense, negation, plural subject, yes/no question, person (I -> she).

1. consistency: changed (slot, new symbol) items per pair; Jaccard within an operation vs between.
2. transplant ("analogy in ink"): learn a consensus edit per operation on 30 pairs (slots that
   change in >= MIN_SHARE of them, set to their most frequent new symbol), apply it to the other
   30 BASE codes, read the patched ring (symbolic), and check:
     exact  = the reading equals the held-out TRANSFORMED sentence
     marker = the operation's marker appears (past form / not / plural / '?' / she)
   Controls: the same number of random slot edits; the consensus edit of a different operation;
   no edit at all.
"""
import collections
import itertools
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

VERBS = [("see", "saw", "sees"), ("eat", "ate", "eats"), ("open", "opened", "opens"), ("watch", "watched", "watches"),
         ("clean", "cleaned", "cleans"), ("buy", "bought", "buys"), ("find", "found", "finds"), ("paint", "painted", "paints"),
         ("love", "loved", "loves"), ("want", "wanted", "wants"), ("break", "broke", "breaks"), ("carry", "carried", "carries")]
OBJS = ["door", "car", "apple", "letter", "window", "garden", "book", "cake", "picture", "box", "kitchen", "bike"]
NOUNS = [("cat", "cats"), ("dog", "dogs"), ("box", "boxes"), ("teacher", "teachers"), ("bird", "birds"), ("car", "cars"),
         ("child", "children"), ("key", "keys"), ("glass", "glasses"), ("letter", "letters"), ("chair", "chairs"), ("boat", "boats")]
PLACES = ["in the kitchen", "near the door", "on the table", "in the garden", "outside", "upstairs"]
MIN_SHARE = 0.4


def pairs(op, rng, n=60):
    out = set()
    while len(out) < n:
        v, o = rng.choice(VERBS), rng.choice(OBJS)
        if op == "past":
            subj = rng.choice(["We", "They", "I", "You"])
            out.add((f"{subj} {v[0]} the {o}.", f"{subj} {v[1]} the {o}."))
        elif op == "negation":
            subj = rng.choice(["We", "They", "I", "You"])
            out.add((f"{subj} {v[0]} the {o}.", f"{subj} don't {v[0]} the {o}."))
        elif op == "plural":
            s, p = rng.choice(NOUNS)
            place = rng.choice(PLACES)
            out.add((f"The {s} is {place}.", f"The {p} are {place}."))
        elif op == "question":
            subj = rng.choice(["you", "they", "we"])
            out.add((f"{subj.capitalize()} can {v[0]} the {o}.", f"Can {subj} {v[0]} the {o}?"))
        elif op == "person":
            out.add((f"I {v[0]} the {o}.", f"She {v[2]} the {o}."))
    return sorted(out)


def marker(op, s):
    s = s.lower()
    return {"past": any(re.search(rf"\b{v[1]}\b", s) for v in VERBS),
            "negation": bool(re.search(r"\bnot\b|n't\b", s)),
            "plural": bool(re.search(r"\bare\b", s)),
            "question": s.strip().endswith("?"),
            "person": s.startswith("she ")}[op]


@torch.no_grad()
def codes(m, texts):
    raw = m.writer_out(texts).float()
    return ((raw[:, :, None, :] - m.books[None]) ** 2).sum(-1).argmin(-1)


@torch.no_grad()
def read(m, sym):
    return m.generate(F.one_hot(m.fsq_to_mark[sym], m.script.n_marks).float())


def main():
    rng = random.Random(0)
    torch.manual_seed(0)
    m, _ = H.load(Path("v4/runs/s2/model.pt"), H.device())
    m.eval()
    ops = ["past", "negation", "plural", "question", "person"]
    data = {}
    for op in ops:
        P = pairs(op, rng)
        rng.shuffle(P)
        a, b = codes(m, [p[0] for p in P]), codes(m, [p[1] for p in P])
        deltas = [{(s, int(b[i, s])) for s in range(H.SLOTS) if a[i, s] != b[i, s]} for i in range(len(P))]
        data[op] = {"pairs": P, "a": a, "b": b, "deltas": deltas}
    res = {"consistency": {}, "transplant": {}}
    for op in ops:
        d = data[op]["deltas"]
        within = [len(x & y) / len(x | y) for x, y in itertools.combinations(d, 2) if x | y]
        between = [len(x & y) / len(x | y) for o2 in ops if o2 != op for x in d[:20] for y in data[o2]["deltas"][:20] if x | y]
        slot_freq = collections.Counter(s for x in d for s, _ in x)
        res["consistency"][op] = {"slots_changed_mean": round(float(np.mean([len(x) for x in d])), 2),
                                  "within_jaccard": round(float(np.mean(within)), 3),
                                  "between_jaccard": round(float(np.mean(between)), 3),
                                  "slots_changing_in_>=60%_of_pairs": sorted(s for s, c in slot_freq.items() if c >= 0.6 * len(d))}
    consensus = {}
    for op in ops:
        d = data[op]["deltas"][:30]
        per_slot = collections.defaultdict(collections.Counter)
        for x in d:
            for s, sym in x:
                per_slot[s][sym] += 1
        consensus[op] = {s: c.most_common(1)[0][0] for s, c in per_slot.items() if sum(c.values()) >= MIN_SHARE * len(d)}
    K = m.books.shape[1]
    for op in ops:
        P, a = data[op]["pairs"][30:], data[op]["a"][30:]
        gold_t = [p[1] for p in P]
        patch = consensus[op]
        other = consensus[ops[(ops.index(op) + 1) % len(ops)]]

        def apply(edits):
            x = a.clone()
            for s, sym in edits.items():
                x[:, s] = sym
            return x

        def rand_edit(k):
            x = a.clone()
            for i in range(len(x)):
                for s in random.Random(i).sample(range(H.SLOTS), k):
                    x[i, s] = (x[i, s] + random.Random(1000 + i * 37 + s).randrange(1, K)) % K
            return x

        conds = {"no_edit": a, "consensus_edit": apply(patch), "random_edit_same_size": rand_edit(max(1, len(patch))),
                 "other_operation_edit": apply(other)}
        row = {"edit_size_slots": len(patch)}
        for name, x in conds.items():
            r = read(m, x)
            row[name] = {"exact_transformed": round(float(np.mean([H.norm(p) == H.norm(g) for p, g in zip(r, gold_t)])), 3),
                         "has_marker": round(float(np.mean([marker(op, p) for p in r])), 3),
                         "example": r[0]}
        row["example_base"], row["example_target"] = P[0][0], P[0][1]
        res["transplant"][op] = row
    Path("v4/runs/s2/grammar_probe.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
