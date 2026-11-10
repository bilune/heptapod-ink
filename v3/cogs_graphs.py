"""COGS logical forms -> canonical meaning graphs, and a clean meaning-level split.

A meaning is a small directed graph:
  nodes: entities {lemma, proper, definite} and events {lemma}
  edges: agent | theme | recipient | xcomp | ccomp | nmod.in | nmod.on | nmod.beside
Variable names (tied to word positions in COGS) are dropped. Nodes get a
canonical order by breadth-first traversal from the root event, children
sorted by relation then lemma, so the same meaning always has the same order
regardless of how the sentence was worded (active/passive, dative alternation).

Splits follow research/notes/meanings_and_survey.md section 3:
  - deduplicate by canonical graph (paraphrases collapse)
  - dev/test drop any graph that also occurs in train
  - gen categories kept per COGS label, plus the audit's semantic relabels
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path

COGS = Path(__file__).resolve().parent.parent / "research" / "repos" / "COGS" / "data"
RELS = ["agent", "theme", "recipient", "xcomp", "ccomp", "nmod.in", "nmod.on", "nmod.beside"]
REL_ORDER = {r: i for i, r in enumerate(RELS)}
ARG = "<arg>"  # unfilled argument of a verb primitive


def parse_lf(lf: str):
    """Return (nodes, edges). nodes: {var: dict}, edges: [(head_var, rel, dep_var)]."""
    lf = lf.replace(" ", "")
    nodes, edges = {}, []
    if lf.startswith("LAMBDA"):
        body = re.sub(r"LAMBDA[a-z]\.", "", lf)
        lf = body
    definite = set()
    if ";" in lf:
        *defs, lf = lf.split(";")
        for d in defs:
            m = re.fullmatch(r"\*(\w+)\((\w+)\)", d)
            nodes[m.group(2)] = {"kind": "ent", "lemma": m.group(1), "proper": False, "definite": True}
    if re.fullmatch(r"[A-Z]\w*", lf):  # primitive proper name, e.g. "Layla"
        return {"c": {"kind": "ent", "lemma": lf, "proper": True, "definite": False}}, []
    for conj in lf.split("AND"):
        m = re.fullmatch(r"(\w+)\((\w+)\)", conj)
        if m:  # unary: noun (may come after the edge that introduced its variable)
            if nodes.get(m.group(2)) is None:
                nodes[m.group(2)] = {"kind": "ent", "lemma": m.group(1), "proper": False, "definite": False}
            continue
        m = re.fullmatch(r"(\w+)\.(\w+)(?:\.(\w+))?\((\w+),(\w+)\)", conj)
        if not m:
            raise ValueError(conj)
        pred, r1, r2, a, b = m.groups()
        rel = f"{r1}.{r2}" if r2 else r1
        if nodes.get(a) is None:  # head may have been introduced earlier as someone's argument
            if rel.startswith("nmod"):
                nodes[a] = {"kind": "ent", "lemma": pred, "proper": False, "definite": False}
            else:
                nodes[a] = {"kind": "ev", "lemma": pred}
        for v in (b,):
            if v not in nodes:
                if re.fullmatch(r"x_\d+|[a-e]", v):
                    nodes[v] = None  # filled later (noun) or placeholder
                else:  # proper name constant
                    nodes[v] = {"kind": "ent", "lemma": v, "proper": True, "definite": False}
        edges.append((a, rel, b))
    for v, n in list(nodes.items()):
        if n is None:
            nodes[v] = {"kind": "ent", "lemma": ARG, "proper": False, "definite": False}
    return nodes, edges


def canonical(nodes, edges):
    """Canonical node order + relabelled edges + a hashable key."""
    children = defaultdict(list)
    has_parent = set()
    for h, r, d in edges:
        children[h].append((REL_ORDER[r], r, d))
        has_parent.add(d)

    def label(v):
        n = nodes[v]
        return (0 if n["kind"] == "ev" else 1, n["lemma"], n.get("proper", False), n.get("definite", False))

    roots = sorted([v for v in nodes if v not in has_parent], key=label)
    order, seen = [], set()
    for root in roots:
        q = deque([root])
        while q:
            v = q.popleft()
            if v in seen:
                continue
            seen.add(v)
            order.append(v)
            for _, _, d in sorted(children[v], key=lambda c: (c[0], label(c[2]))):
                q.append(d)
    idx = {v: i for i, v in enumerate(order)}
    cnodes = [nodes[v] for v in order]
    cedges = sorted((idx[h], r, idx[d]) for h, r, d in edges)
    key = json.dumps([[label(v) for v in order], cedges])
    return cnodes, cedges, key


def load(split: str):
    rows = []
    for line in (COGS / f"{split}.tsv").read_text().splitlines():
        sent, lf, cat = line.split("\t")
        nodes, edges = parse_lf(lf)
        cn, ce, key = canonical(nodes, edges)
        rows.append({"sentence": sent, "category": cat, "nodes": cn, "edges": ce, "key": key})
    return rows


def build(max_nodes: int = 12):
    train = load("train")
    seen = {}
    for r in train:
        seen.setdefault(r["key"], r)
    out = {"train": list(seen.values())}
    for split in ("dev", "test", "gen"):
        rows, keys = [], set()
        for r in load(split):
            if r["key"] in seen or r["key"] in keys:
                continue
            keys.add(r["key"])
            rows.append(r)
        out[split] = rows
    # semantic relabel from the audit: dative alternations are not new meanings
    for r in out["gen"]:
        if r["category"] in ("do_dative_to_pp_dative", "pp_dative_to_do_dative"):
            r["category"] = "rare_verb_fillers"
    stats = {}
    for k, v in out.items():
        fits = [r for r in v if len(r["nodes"]) <= max_nodes]
        stats[k] = {"meanings": len(v), f"<= {max_nodes} nodes": len(fits),
                    "max_nodes": max(len(r["nodes"]) for r in v)}
    lemmas = sorted({n["lemma"] for v in out.values() for r in v for n in r["nodes"]})
    return out, stats, lemmas


if __name__ == "__main__":
    data, stats, lemmas = build()
    print(json.dumps(stats, indent=1))
    print("lemmas", len(lemmas))
    ex = data["train"][5]
    print(ex["sentence"], ex["nodes"], ex["edges"], sep="\n")
    cats = defaultdict(int)
    for r in data["gen"]:
        cats[r["category"]] += 1
    print(dict(cats))
    out = Path(__file__).resolve().parent / "data"
    out.mkdir(exist_ok=True)
    for k, v in data.items():
        (out / f"{k}.json").write_text(json.dumps([{x: r[x] for x in ("sentence", "category", "nodes", "edges")} for r in v]))
    (out / "lemmas.json").write_text(json.dumps(lemmas))
