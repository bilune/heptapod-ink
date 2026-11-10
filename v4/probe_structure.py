#!/usr/bin/env python3
"""Is there word structure inside the ring? Two probes on the frozen system (plain writer, no search).

1. swap   For a word pair (w1 -> w2), take sentences containing w1 as a token, replace it
          with w2, and compare the two codes. A word with its own "spelling" should change
          the same slots to the same marks across sentences (high within-pair Jaccard of the
          changed (slot, mark) items, vs. between pairs). Also: is w2 read back correctly?
2. poke   Replace one slot's symbol by another codebook entry and read the ring (symbolic
          path; the eyes read 99% of marks). How many words change, and where in the sentence?
          If slots specialise, a slot should keep hitting the same relative position.

  .venv/bin/python v4/probe_structure.py --ckpt v4/runs/s2/model.pt
"""
from __future__ import annotations

import argparse
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

PAIRS = [("dog", "cat"), ("house", "car"), ("Tom", "Mary"), ("book", "letter"), ("two", "three"),
         ("happy", "sad"), ("eat", "drink"), ("went", "came"), ("today", "yesterday"), ("Paris", "London")]


@torch.no_grad()
def codes(m, texts, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        raw = m.writer_out(texts[i:i + bs]).float()
        out.append(((raw[:, :, None, :] - m.books[None]) ** 2).sum(-1).argmin(-1).cpu())   # symbols (B, S)
    return torch.cat(out)


@torch.no_grad()
def read_symbols(m, sym, bs=64):
    preds = []
    for i in range(0, len(sym), bs):
        marks = m.fsq_to_mark[sym[i:i + bs].to(m.dev)]
        preds += m.generate(F.one_hot(marks, m.script.n_marks).float())
    return preds


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def swap_probe(m, corpus, per_pair, rng):
    report, deltas = {}, {}
    for w1, w2 in PAIRS:
        pat = re.compile(rf"\b{re.escape(w1)}\b")
        hits = [s for s in corpus if len(pat.findall(s)) == 1 and not re.search(rf"\b{re.escape(w2)}\b", s)]
        hits = rng.sample(hits, min(per_pair, len(hits)))
        if len(hits) < 10:
            continue
        swapped = [pat.sub(w2, s) for s in hits]
        a, b = codes(m, hits), codes(m, swapped)
        d = [{(s_, int(b[i, s_])) for s_ in range(H.SLOTS) if a[i, s_] != b[i, s_]} for i in range(len(hits))]
        slots = [{s_ for s_, _ in x} for x in d]
        deltas[(w1, w2)] = d
        reads = read_symbols(m, b)
        within = [jac(d[i], d[j]) for i, j in itertools.combinations(range(len(d)), 2)]
        within_slots = [jac(slots[i], slots[j]) for i, j in itertools.combinations(range(len(d)), 2)]
        freq = collections.Counter(s_ for x in slots for s_ in x)
        top5 = sum(c for _, c in freq.most_common(5)) / max(1, sum(freq.values()))
        report[f"{w1}->{w2}"] = {
            "sentences": len(hits),
            "slots_changed_mean": round(float(np.mean([len(x) for x in d])), 2),
            "within_jaccard_items": round(float(np.mean(within)), 3),
            "within_jaccard_slots": round(float(np.mean(within_slots)), 3),
            "top5_slots_share_of_changes": round(top5, 3),
            "reads_new_word": round(float(np.mean([bool(re.search(rf"\b{re.escape(w2)}\b", r)) for r in reads])), 3),
            "reads_exact": round(float(np.mean([H.norm(r) == H.norm(t) for r, t in zip(reads, swapped)])), 3),
        }
    keys = list(deltas)
    between = []
    for _ in range(3000):
        k1, k2 = rng.sample(keys, 2)
        between.append(jac(rng.choice(deltas[k1]), rng.choice(deltas[k2])))
    summary = {"between_pairs_jaccard_items": round(float(np.mean(between)), 3),
               "within_pairs_jaccard_items_mean": round(float(np.mean([v["within_jaccard_items"] for v in report.values()])), 3),
               "slots_changed_mean": round(float(np.mean([v["slots_changed_mean"] for v in report.values()])), 2)}
    return report, summary


def poke_probe(m, corpus, n_sent, rng):
    texts = rng.sample(corpus, n_sent)
    base = codes(m, texts)
    base_read = read_symbols(m, base)
    K = m.books.shape[1]
    rows = []                                              # (slot, words_changed, n_words, rel positions)
    for s_ in range(H.SLOTS):
        poked = base.clone()
        poked[:, s_] = (base[:, s_] + torch.randint(1, K, (len(texts),))) % K
        reads = read_symbols(m, poked)
        for t0, r in zip(base_read, reads):
            a, b = H.norm(t0).split(), H.norm(r).split()
            changed = [i for i in range(max(len(a), len(b))) if i >= len(a) or i >= len(b) or a[i] != b[i]]
            rows.append((s_, len(changed), len(a), [i / max(1, len(a) - 1) for i in changed if i < len(a)]))
    by_slot = collections.defaultdict(list)
    for s_, c, n, pos in rows:
        by_slot[s_].append((c, n, pos))
    per_slot = {}
    for s_, v in by_slot.items():
        pos = [p for _, _, ps in v for p in ps]
        per_slot[s_] = {"no_change": round(float(np.mean([c == 0 for c, _, _ in v])), 3),
                        "words_changed_mean": round(float(np.mean([c for c, _, _ in v])), 2),
                        "rel_position_mean": round(float(np.mean(pos)), 3) if pos else None,
                        "rel_position_std": round(float(np.std(pos)), 3) if pos else None}
    allc = [c for _, c, _, _ in rows]
    alln = [n for _, _, n, _ in rows]
    summary = {"pokes": len(rows), "no_change": round(float(np.mean([c == 0 for c in allc])), 3),
               "one_word_changed": round(float(np.mean([c == 1 for c in allc])), 3),
               "half_or_more_words_changed": round(float(np.mean([c >= n / 2 for c, n in zip(allc, alln)])), 3),
               "position_std_within_slot_mean": round(float(np.mean([v["rel_position_std"] for v in per_slot.values() if v["rel_position_std"] is not None])), 3),
               "position_std_uniform_reference": round(float(np.sqrt(1 / 12)), 3)}
    # heatmap: slot x relative-position bin of changed words
    heat = np.zeros((H.SLOTS, 10))
    for s_, _, _, pos in rows:
        for p in pos:
            heat[s_, min(9, int(p * 10))] += 1
    return per_slot, summary, heat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="v4/runs/s2/model.pt")
    ap.add_argument("--per-pair", type=int, default=50)
    ap.add_argument("--poke-sentences", type=int, default=120)
    ap.add_argument("--out", default="v4/runs/s2/probe_structure.json")
    args = ap.parse_args()
    rng = random.Random(0)
    torch.manual_seed(0)
    m, _ = H.load(Path(args.ckpt), H.device())
    m.eval()
    corpus = H.load_rows("train")                                    # natural Tatoeba sentences (common words)
    swap, swap_sum = swap_probe(m, corpus, args.per_pair, rng)
    print(json.dumps({"swap_summary": swap_sum, "swap": swap}, indent=1), flush=True)
    per_slot, poke_sum, heat = poke_probe(m, H.load_rows("dev"), args.poke_sentences, rng)
    print(json.dumps({"poke_summary": poke_sum}, indent=1), flush=True)
    Path(args.out).write_text(json.dumps({"swap_summary": swap_sum, "swap": swap, "poke_summary": poke_sum,
                                          "poke_per_slot": per_slot}, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    heat = heat / np.maximum(heat.sum(1, keepdims=True), 1)
    plt.figure(figsize=(6, 8))
    plt.imshow(heat, aspect="auto", cmap="Blues")
    plt.xlabel("relative position of the changed word in the sentence (0 = start, 1 = end)")
    plt.ylabel("ring slot that was changed (0-35)")
    plt.xticks(range(10), [f"{i / 10:.1f}" for i in range(10)], fontsize=7)
    plt.colorbar(label="fraction of that slot's changes")
    plt.title("Does each ring slot affect a fixed part of the sentence?", fontsize=10)
    plt.tight_layout()
    plt.savefig(Path(args.out).with_suffix(".png"), dpi=110)
    print(Path(args.out).with_suffix(".png"))


if __name__ == "__main__":
    main()
