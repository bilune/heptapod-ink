#!/usr/bin/env python3
"""Build the mixed natural-English training pool (debate 4, round 2).

Sources (streamed, bounded; nothing downloaded in full):
  FineWeb sample-10BT (ODC-By + Common Crawl terms)   55%
  Wikipedia 20231101.en (CC BY-SA / GFDL)             15%
  CC-News via vblagoje/cc_news                        10%
  Tatoeba train replay (CC BY 2.0)                    20%
(DailyDialog, proposed by Codex, no longer loads with the current `datasets`; omitted.)

Per source: split documents into sentences, keep 2-12 words, at most 5 per document,
half drawn uniformly and half from "hard" sentences (digits, internal capitals,
or quotes/brackets/colons). Exclusions: every UD English sentence we hold, every
Tatoeba string, every evaluation panel (exact and normalized match), plus near
duplicates of evaluation sentences (character 5-gram Jaccard >= 0.85).
T5 round-trip clean and <= 64 T5 tokens. 5,000 document-disjoint validation sentences.

  .venv/bin/python v4/prep_mixed.py --total 500000
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v4"))
import heptapod_t5 as H  # noqa: E402

SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])|\n+")
BAD = re.compile(r"https?://|www\.|@|\|{1}|\{|\}|<|>|#|\.com\b|©|\.\.\.\.|_{2,}")
HARD = re.compile(r"\d|\s[A-Z]|[\"'():;\[\]]")


def norm(s):
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def grams(s, n=5):
    s = f" {norm(s)} "
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


class Excluder:
    def __init__(self):
        exact, evals = set(), []
        for line in (ROOT / "research" / "tatoeba" / "eng_sentences.tsv").read_text(encoding="utf-8").splitlines():
            p = line.split("\t")
            if len(p) >= 3:
                exact.add(norm(p[2]))
        for d in ("ud_ewt", "ud_locked"):
            for f in (ROOT / "research" / d).glob("*.conllu"):
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.startswith("# text = "):
                        s = " ".join(line[9:].split())
                        exact.add(norm(s))
                        if len(s.split()) <= 14:
                            evals.append(s)
        for f in H.DATA.glob("*.json"):
            if f.name in ("train.json", "random_vocab.json") or f.name.startswith("mixed_"):
                continue
            for r in json.loads(f.read_text()):
                exact.add(norm(r["sentence"]))
                if not f.name.endswith("random.json") and "uniform" not in f.name:
                    evals.append(r["sentence"])
        self.exact = exact
        self.eval_grams = [grams(s) for s in evals]
        index = collections.defaultdict(list)
        for i, g in enumerate(self.eval_grams):
            for x in g:
                index[x].append(i)
        self.index = {k: v for k, v in index.items() if len(v) <= 300}  # rare grams find candidates
        print({"exclusion_exact": len(exact), "eval_sentences_for_jaccard": len(evals)}, flush=True)

    def bad(self, s):
        if norm(s) in self.exact:
            return "exact"
        g = grams(s)
        cnt = collections.Counter()
        for x in g:
            for i in self.index.get(x, ()):
                cnt[i] += 1
        for i, c in cnt.most_common(5):
            e = self.eval_grams[i]
            if c >= 0.5 * len(g) and len(g & e) / len(g | e) >= 0.85:
                return "near"
        return None


def sentences(text):
    for s in SPLIT.split(text):
        s = " ".join(s.split())
        n = len(s.split())
        if 2 <= n <= 12 and not BAD.search(s) and s[:1].isupper() and sum(c.isalpha() for c in s) >= 0.6 * len(s):
            if s.isascii() or all(ord(c) < 0x2000 for c in s):
                yield s


def stream(name):
    from datasets import load_dataset
    if name == "fineweb":
        return (r["text"] for r in load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True))
    if name == "wikipedia":
        return (r["text"] for r in load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True))
    if name == "news":
        return (r["text"] for r in load_dataset("vblagoje/cc_news", split="train", streaming=True))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=500000)
    ap.add_argument("--val", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(H.T5)
    ex = Excluder()
    quotas = {"fineweb": 0.55, "wikipedia": 0.15, "news": 0.10}
    seen = set()
    train, val, stats = [], [], collections.Counter()
    for src, frac in quotas.items():
        want = int(args.total * frac)
        want_val = int(args.val * frac / sum(quotas.values()))
        easy, hard, vbuf = [], [], []
        t0, docs = time.time(), 0
        for text in stream(src):
            docs += 1
            to_val = docs % 50 == 0 and len(vbuf) < want_val          # document-disjoint validation
            cands = list(sentences(text))
            rng.shuffle(cands)
            kept = 0
            for s in cands:
                if kept >= 5:
                    break
                k = norm(s)
                if k in seen:
                    stats["dup"] += 1
                    continue
                why = ex.bad(s)
                if why:
                    stats[why] += 1
                    continue
                ids = tok(s).input_ids
                if len(ids) > 64 or tok.decode(ids, skip_special_tokens=True) != s:
                    stats["tokenizer"] += 1
                    continue
                seen.add(k)
                kept += 1
                if to_val:
                    vbuf.append(s)
                else:
                    (hard if HARD.search(s[1:]) else easy).append(s)
            if docs % 2000 == 0:
                print({"src": src, "docs": docs, "easy": len(easy), "hard": len(hard), "val": len(vbuf),
                       "sec": round(time.time() - t0)}, flush=True)
            if len(easy) >= want // 2 and len(hard) >= want // 2 and len(vbuf) >= want_val:
                break
        half = want // 2
        pick = rng.sample(easy, min(half, len(easy))) + rng.sample(hard, min(want - min(half, len(easy)), len(hard)))
        train += [{"sentence": s, "category": src} for s in pick]
        val += [{"sentence": s, "category": src} for s in vbuf[:want_val]]
        stats[f"{src}_train"] = len(pick)
        stats[f"{src}_docs"] = docs
    tat = H.load_rows("train")
    for s in rng.sample(tat, int(args.total * 0.20)):
        train.append({"sentence": s, "category": "tatoeba"})
    rng.shuffle(train)
    (H.DATA / "mixed_train.json").write_text(json.dumps(train))
    (H.DATA / "mixed_val.json").write_text(json.dumps(val))
    report = {"train": len(train), "val": len(val), **stats,
              "examples": [r["sentence"] for r in train[:12]]}
    (ROOT / "v4" / "mixed_data_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
