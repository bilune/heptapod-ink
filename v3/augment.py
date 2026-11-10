#!/usr/bin/env python3
"""Train-only lexical substitution augmentation for the reader.

Every rare noun or name should be seen in many different sentences.
A word only replaces another in a role (agent, theme, recipient, ...) where
it was itself attested in train, so animacy-like restrictions are kept. We take
deduplicated COGS train sentences and swap ONE entity word for another of the
same kind (common noun <-> common noun, proper name <-> proper name), with the
replacement sampled in proportion to how far that word is below `--target`
training contexts. Everything (vocabulary, frequencies, templates) comes from
train only. With --verbs, a verb can also be swapped for another verb attested in train
with the same argument frame and the same surface shape (past, was+participle,
to+infinitive), only in single-verb sentences. Articles and prepositions are
left as they are.

Held-out protection: after generation, any variant whose normalized text or
canonical meaning graph equals a dev/test/gen item is dropped; only the number
dropped is reported, nothing else from held-out data is used.

Note: the noun/name distinction comes from the COGS meaning graphs. This is
information used to build practice data, not a rule inside the model.

  python3 v3/augment.py --max 150000 --target 200
"""
import argparse
import collections
import json
import random
from pathlib import Path

import cogs_graphs as C

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def norm(s):
    return " ".join(s.lower().replace(" .", ".").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=150000)
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbs", type=int, default=0, help="extra variants with a verb swapped")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    train = json.loads((DATA / "train.json").read_text())
    kinds = {"common": set(), "proper": set()}
    contexts = collections.Counter()
    role_pool = collections.defaultdict(set)  # (kind, role) -> lemmas attested in that role in train

    def role_of(r, j):
        rels = sorted({rel for h, rel, d in r["edges"] if d == j})
        return rels[0] if rels else "none"

    for r in train:
        for j, n in enumerate(r["nodes"]):
            if n["kind"] == "ent" and n["lemma"] != C.ARG:
                role_pool[("proper" if n.get("proper") else "common", role_of(r, j))].add(n["lemma"])
    for r in train:
        seen = set()
        for n in r["nodes"]:
            if n["kind"] != "ent" or n["lemma"] == C.ARG:
                continue
            kinds["proper" if n.get("proper") else "common"].add(n["lemma"])
            seen.add(n["lemma"])
        contexts.update(seen)

    def weights(pool):
        lem = sorted(pool)
        w = [max(args.target - contexts[l], 0) + 1 for l in lem]
        return lem, w

    pools = {k: weights(v) for k, v in role_pool.items()}

    # slots that can be swapped: an entity lemma appearing exactly once as a token
    jobs = []
    for i, r in enumerate(train):
        toks = r["sentence"].split()
        for j, n in enumerate(r["nodes"]):
            if n["kind"] != "ent" or n["lemma"] == C.ARG:
                continue
            if toks.count(n["lemma"]) == 1 and sum(m.get("lemma") == n["lemma"] for m in r["nodes"]) == 1:
                jobs.append((i, j, ("proper" if n.get("proper") else "common", role_of(r, j))))

    held_text, held_keys = set(), set()
    for split in ("dev", "test", "gen"):
        for r in json.loads((DATA / f"{split}.json").read_text()):
            held_text.add(norm(r["sentence"]))
            held_keys.add(C.canonical({k: v for k, v in enumerate(r["nodes"])},
                                      [tuple(e) for e in r["edges"]])[2])
    train_text = {norm(r["sentence"]) for r in train}

    out, seen_text = [], set()
    dropped_heldout = dropped_dup = 0
    attempts = 0
    while len(out) < args.max and attempts < args.max * 6:
        attempts += 1
        i, j, kind = rng.choice(jobs)
        r = train[i]
        old = r["nodes"][j]["lemma"]
        lem, w = pools[kind]
        new = rng.choices(lem, weights=w)[0]
        if new == old or any(n.get("lemma") == new for n in r["nodes"]):
            continue
        toks = r["sentence"].split()
        toks[toks.index(old)] = new
        sent = " ".join(toks)
        t = norm(sent)
        if t in seen_text or t in train_text:
            dropped_dup += 1
            continue
        nodes = [dict(n) for n in r["nodes"]]
        nodes[j]["lemma"] = new
        key = C.canonical(dict(enumerate(nodes)), [tuple(e) for e in r["edges"]])[2]
        if t in held_text or key in held_keys:
            dropped_heldout += 1
            continue
        seen_text.add(t)
        out.append({"sentence": sent, "category": "aug", "swapped": [old, new]})

    # ---- verbs: swap a verb for another attested (in train) with the same
    # argument frame AND the same surface shape (past / was+participle / to+infinitive)
    FUNC = {"a", "the", "was", "by", "to", "that", ".", "in", "on", "beside"}
    ent_words = {l for l in kinds["common"] | kinds["proper"]}
    forms = collections.defaultdict(dict)        # lemma -> {shape: surface}
    frames = collections.defaultdict(set)        # (frame, shape) -> lemmas
    verb_jobs = []

    def shape_of(toks, k):
        prev = toks[k - 1] if k > 0 else ""
        return "participle" if prev == "was" else "infinitive" if prev == "to" else "past"

    for i, r in enumerate(train):
        events = [j for j, n in enumerate(r["nodes"]) if n["kind"] == "ev"]
        if len(events) != 1:
            continue
        j = events[0]
        toks = r["sentence"].split()
        cand = [k for k, t in enumerate(toks) if t not in FUNC and t not in ent_words]
        if len(cand) != 1:
            continue
        k = cand[0]
        lemma, shape = r["nodes"][j]["lemma"], shape_of(toks, k)
        frame = tuple(sorted({rel for h, rel, d in r["edges"] if h == j}))
        forms[lemma][shape] = toks[k]
        frames[(frame, shape)].add(lemma)
        verb_jobs.append((i, j, k, frame, shape))
    verb_out = 0
    tries = 0
    while verb_out < args.verbs and tries < args.verbs * 8 and verb_jobs:
        tries += 1
        i, j, k, frame, shape = rng.choice(verb_jobs)
        r = train[i]
        pool = sorted(l for l in frames[(frame, shape)] if l != r["nodes"][j]["lemma"] and shape in forms[l])
        if not pool:
            continue
        new = rng.choice(pool)
        toks = r["sentence"].split()
        toks[k] = forms[new][shape]
        sent = " ".join(toks)
        t = norm(sent)
        if t in seen_text or t in train_text:
            continue
        nodes = [dict(n) for n in r["nodes"]]
        nodes[j]["lemma"] = new
        key = C.canonical(dict(enumerate(nodes)), [tuple(e) for e in r["edges"]])[2]
        if t in held_text or key in held_keys:
            dropped_heldout += 1
            continue
        seen_text.add(t)
        out.append({"sentence": sent, "category": "aug_verb", "swapped": [r["nodes"][j]["lemma"], new]})
        verb_out += 1

    after = collections.Counter(contexts)
    for o in out:
        after[o["swapped"][1]] += 1
    all_lem = kinds["common"] | kinds["proper"]
    report = {
        "variants": len(out), "verb_variants": verb_out, "attempts": attempts,
        "dropped_matching_heldout": dropped_heldout, "dropped_duplicates": dropped_dup,
        "lemmas": len(all_lem),
        "lemmas_below_target_before": sum(contexts[l] < args.target for l in all_lem),
        "lemmas_below_target_after": sum(after[l] < args.target for l in all_lem),
        "median_contexts_before": sorted(contexts[l] for l in all_lem)[len(all_lem) // 2],
        "median_contexts_after": sorted(after[l] for l in all_lem)[len(all_lem) // 2],
        "examples": [o["sentence"] for o in out[:4]] + [o["sentence"] for o in out if o["category"] == "aug_verb"][:6],
    }
    (DATA / "train_aug.json").write_text(json.dumps(out))
    (DATA / "train_aug_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
