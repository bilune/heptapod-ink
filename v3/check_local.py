#!/usr/bin/env python3
"""Two free local checks on a trained checkpoint, before spending on Modal.

  consistency  Does the frozen writer draw a word the same way across sentences?
               For a word w and a context sentence c, delta(c, w) is the set of
               (slot, head, choice) items present when w replaces a fixed anchor
               word in c but absent with the anchor. If w has its own stable
               "spelling", its deltas overlap across contexts (within-word
               Jaccard) much more than deltas of different words do
               (between-word Jaccard). Compared for rare vs frequent words.
               Uses train sentences and train vocabulary only.

  verify       Beam search + analysis-by-synthesis on dev: the reader proposes
               candidate sentences, the frozen writer re-draws each one, and a
               candidate whose drawing matches the received image is preferred.
               Inputs are only the image and the reader's own candidates.

  .venv/bin/python v3/check_local.py <checkpoint_dir> consistency
  .venv/bin/python v3/check_local.py <checkpoint_dir> verify --n 1500 --beam 8
"""
import argparse
import collections
import itertools
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod as V  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


@torch.no_grad()
def codes(model, texts):
    out = []
    for i in range(0, len(texts), 128):
        rc, mc = model.writer(*model.read_text(model.tokenize(texts[i:i + 128])))
        out.append(torch.stack([rc.argmax(-1), mc.argmax(-1)], -1).cpu())
    return torch.cat(out)  # (N, slots, 2)


def items(code):
    return {(s, h, int(code[s, h])) for s in range(code.shape[0]) for h in range(2)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def consistency(model, args):
    rows = json.loads((DATA / "train.json").read_text())
    freq = collections.Counter()
    common = set()
    for r in rows:
        for n in r["nodes"]:
            if n["kind"] == "ent" and not n.get("proper") and n["lemma"] != "<arg>":
                common.add(n["lemma"])
        freq.update(V.normalize(r["sentence"]).replace(".", " .").split())
    anchor = "cake"
    # contexts: train sentences where the anchor is a theme, single mention
    ctx = []
    for r in rows:
        toks = r["sentence"].split()
        if toks.count(anchor) != 1:
            continue
        for h, rel, d in r["edges"]:
            if r["nodes"][d]["lemma"] == anchor and rel == "theme":
                ctx.append(r["sentence"])
                break
    rng = random.Random(0)
    ctx = rng.sample(ctx, min(args.contexts, len(ctx)))
    buckets = {"1-9": (1, 9), "10-99": (10, 99), "100-999": (100, 999), "1000+": (1000, 10**9)}
    picks = {}
    for name, (lo, hi) in buckets.items():
        cand = sorted(w for w in common if lo <= freq[w] <= hi and w != anchor and " " not in w)
        picks[name] = rng.sample(cand, min(args.words, len(cand)))
    base = codes(model, ctx)
    deltas = {}
    for name, words in picks.items():
        for w in words:
            texts = [c.replace(f" {anchor} ", f" {w} ").replace(f" {anchor} .", f" {w} .") for c in ctx]
            cw = codes(model, texts)
            deltas[w] = [items(cw[i]) - items(base[i]) for i in range(len(ctx))]
    report = {"anchor": anchor, "contexts": len(ctx), "words_per_bucket": args.words, "buckets": {}}
    all_words = [w for ws in picks.values() for w in ws]
    for name, words in picks.items():
        within, between, size = [], [], []
        for w in words:
            d = deltas[w]
            size += [len(x) for x in d]
            within += [jaccard(d[i], d[j]) for i, j in itertools.combinations(range(len(d)), 2)]
            others = [o for o in all_words if o != w]
            for i in range(len(d)):
                o = rng.choice(others)
                between.append(jaccard(d[i], deltas[o][i]))  # same context, different word
        report["buckets"][name] = {
            "within_word_jaccard": round(sum(within) / len(within), 3),
            "between_word_jaccard_same_context": round(sum(between) / len(between), 3),
            "avg_items_changed": round(sum(size) / len(size), 1),
            "example_words": words[:6],
        }
    print(json.dumps(report, indent=1))
    (V.OUT / "check_consistency.json").write_text(json.dumps(report, indent=1))


@torch.no_grad()
def beam_search(reader, img, bos, eos, pad, beam, max_len=V.MAX_TOKENS):
    """Batched beam search. Returns (B, beam, T) token ids and (B, beam) log-probs."""
    B = img.shape[0]
    mem = reader.memory(img).repeat_interleave(beam, 0)
    seq = torch.full((B * beam, 1), bos, dtype=torch.long, device=img.device)
    score = torch.zeros(B, beam, device=img.device)
    score[:, 1:] = -1e9  # start from a single live beam
    done = torch.zeros(B * beam, dtype=torch.bool, device=img.device)
    for _ in range(max_len - 1):
        t = seq.shape[1]
        x = reader.tok(seq) + reader.tpos[:t]
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=img.device), 1)
        logp = F.log_softmax(reader.out(reader.dec(x, mem, tgt_mask=causal))[:, -1], -1)
        V_ = logp.shape[-1]
        # finished beams can only extend with pad at no cost
        logp = torch.where(done[:, None], torch.full_like(logp, -1e9), logp)
        logp[done, pad] = 0.0
        total = (score.view(-1, 1) + logp).view(B, beam * V_)
        top, idx = total.topk(beam, -1)
        src = idx // V_
        tok = idx % V_
        base = (torch.arange(B, device=img.device) * beam)[:, None]
        sel = (base + src).view(-1)
        seq = torch.cat([seq[sel], tok.view(-1, 1)], 1)
        done = done[sel] | (tok.view(-1) == eos)
        score = top
        if done.all():
            break
    return seq.view(B, beam, -1), score


def verify(model, args):
    rows = json.loads((DATA / f"{args.split}.json").read_text())[: args.n]
    texts = [r["sentence"] for r in rows]
    gold = [V.normalize(t) for t in texts]
    stats = collections.Counter()
    for i in range(0, len(texts), args.batch):
        chunk, g = texts[i:i + args.batch], gold[i:i + args.batch]
        with torch.no_grad():
            rc, mc = model.writer(*model.read_text(model.tokenize(chunk)))
            img = model.script(rc, mc)
            greedy = model.decode(model.reader.generate(img, model.bos, model.eos))
            seqs, scores = beam_search(model.reader, img, model.bos, model.eos, model.pad, args.beam)
            cands = [[model.decode(s[None])[0] for s in seqs[b]] for b in range(len(chunk))]
            # re-draw every distinct candidate with the frozen writer
            flat = sorted({c for cs in cands for c in cs} | set(greedy))
            redraw = {}
            for j in range(0, len(flat), 128):
                part = flat[j:j + 128]
                r2, m2 = model.writer(*model.read_text(model.tokenize(part)))
                for c, im in zip(part, model.script(r2, m2)):
                    redraw[c] = im
        for b in range(len(chunk)):
            pool = list(dict.fromkeys(cands[b] + [greedy[b]]))
            match = [c for c in pool if (redraw[c] - img[b]).abs().max().item() < args.tol]
            beam_top = cands[b][0]
            if match:
                ranked = [c for c in cands[b] if c in match] or match
                chosen = ranked[0]
            else:
                chosen = greedy[b]
            stats["greedy"] += greedy[b] == g[b]
            stats["beam_top1"] += beam_top == g[b]
            stats["gold_in_candidates"] += g[b] in pool
            stats["verified"] += chosen == g[b]
            stats["any_match"] += bool(match)
            stats["multiple_matches"] += len(match) > 1
            stats["false_match"] += bool(match) and chosen != g[b]
        print({k: round(v / min(i + args.batch, len(texts)), 4) for k, v in stats.items()}, flush=True)
    n = len(texts)
    report = {"split": args.split, "n": n, "beam": args.beam, **{k: round(v / n, 4) for k, v in stats.items()}}
    print(json.dumps(report, indent=1))
    (V.OUT / f"check_verify_{args.split}.json").write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("what", choices=["consistency", "verify"])
    ap.add_argument("--contexts", type=int, default=20)
    ap.add_argument("--words", type=int, default=12)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    args = ap.parse_args()
    V.OUT = Path(args.ckpt_dir).resolve()
    model = V.load(V.device())
    model.eval()
    {"consistency": consistency, "verify": verify}[args.what](model, args)


if __name__ == "__main__":
    main()
