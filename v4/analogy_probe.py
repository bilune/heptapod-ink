#!/usr/bin/env python3
"""Analogies in ink: b - a + c ~ d, computed on the writer's continuous slot vectors.

For each analogy (a : b :: c : d), e.g. china : chinese :: america : american:
  z = raw(b) - raw(a) + raw(c)        (36 slot vectors, before choosing marks)
  ring(z) = nearest codebook symbol per slot -> marks -> reader (symbolic and through the image)
Measures: the reader outputs d; Hamming(ring(z), ring(d)) vs Hamming(ring(c), ring(d)).
Controls: the same arithmetic with the offset taken from a DIFFERENT relation family
(b' - a' + c), and no offset (just ring(c)).
Words are encoded inside a fixed carrier sentence ("The answer was {w}.") because the system was
trained on sentences; success = the expected word d appears in the reading (and c does not).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402

FAMILIES = {
    "country->nationality": [("China", "Chinese"), ("America", "American"), ("France", "French"), ("Italy", "Italian"),
                             ("Japan", "Japanese"), ("Germany", "German"), ("Spain", "Spanish"), ("Mexico", "Mexican"),
                             ("Russia", "Russian"), ("Brazil", "Brazilian")],
    "male->female": [("king", "queen"), ("man", "woman"), ("boy", "girl"), ("father", "mother"), ("brother", "sister"),
                     ("uncle", "aunt"), ("son", "daughter"), ("husband", "wife"), ("prince", "princess"), ("actor", "actress")],
    "present->past": [("walk", "walked"), ("see", "saw"), ("eat", "ate"), ("go", "went"), ("write", "wrote"),
                      ("play", "played"), ("run", "ran"), ("buy", "bought"), ("think", "thought"), ("open", "opened")],
    "singular->plural": [("cat", "cats"), ("dog", "dogs"), ("child", "children"), ("box", "boxes"), ("city", "cities"),
                         ("house", "houses"), ("mouse", "mice"), ("car", "cars"), ("woman", "women"), ("book", "books")],
    "adjective->comparative": [("big", "bigger"), ("small", "smaller"), ("fast", "faster"), ("good", "better"),
                               ("old", "older"), ("long", "longer"), ("cold", "colder"), ("happy", "happier"),
                               ("young", "younger"), ("strong", "stronger")],
}


CARRIER = "The answer was {}."


@torch.no_grad()
def raw(m, words):
    return m.writer_out([CARRIER.format(w) for w in words]).float()   # (N, S, D)


def has(reading, w):
    import re
    return bool(re.search(rf"\b{re.escape(w.lower())}\b", reading.lower()))


@torch.no_grad()
def quantize(m, z):
    return ((z[:, :, None, :] - m.books[None]) ** 2).sum(-1).argmin(-1)


@torch.no_grad()
def read(m, sym, visual):
    marks = m.fsq_to_mark[sym]
    probs = m.eye_probs(m.draw(marks)) if visual else F.one_hot(marks, m.script.n_marks).float()
    return m.generate(probs)


def main():
    m, _ = H.load(Path("v4/runs/s2/model.pt"), H.device())
    m.eval()
    rng = random.Random(0)
    fams = list(FAMILIES)
    words = sorted({w for f in FAMILIES.values() for p in f for w in p})
    R = dict(zip(words, raw(m, words)))
    Q = dict(zip(words, quantize(m, torch.stack([R[w] for w in words]))))
    plain_read = dict(zip(words, read(m, torch.stack([Q[w] for w in words]), True)))
    res, examples = {}, []
    for fam, pairs in FAMILIES.items():
        rows = {"analogy": [], "other_offset": [], "no_offset": []}
        for (a, b), (c, d) in [(p, q) for p in pairs for q in pairs if p != q]:
            other_fam = rng.choice([f for f in fams if f != fam])
            a2, b2 = rng.choice(FAMILIES[other_fam])
            zs = {"analogy": R[b] - R[a] + R[c], "other_offset": R[b2] - R[a2] + R[c], "no_offset": R[c]}
            for k, z in zs.items():
                rows[k].append((z, d, f"{b} - {a} + {c}" if k == "analogy" else k))
        out = {}
        for k, items in rows.items():
            sym = quantize(m, torch.stack([z for z, _, _ in items]))
            gold = torch.stack([Q[d] for _, d, _ in items])
            ham = (sym != gold).sum(-1).float()
            for visual in (False, True):
                reads = read(m, sym, visual)
                hit = [has(r, d) for r, (_, d, _) in zip(reads, items)]
                out[f"{k}_{'image' if visual else 'symbolic'}_reads_d"] = round(float(np.mean(hit)), 3)
            out[f"{k}_hamming_to_ring_d"] = round(float(ham.mean()), 2)
            if k == "analogy":
                for (_, d, label), r in list(zip(items, reads))[:6]:
                    examples.append(f"{label} -> reads '{r}' (expected {d})")
        out["n"] = len(rows["analogy"])
        res[fam] = out
    res["direct_rings_read_back_correctly"] = round(float(np.mean([H.norm(plain_read[w]) == H.norm(CARRIER.format(w)) for w in words])), 3)
    res["examples"] = examples
    Path("v4/runs/s2/analogy_probe.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
