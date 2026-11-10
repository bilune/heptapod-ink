#!/usr/bin/env python3
"""Sender-side search ("the writer rereads before sending").

The writer knows the sentence it wants to send. Starting from its own code, it
tries mark substitutions and keeps the ring that the reader understands best.
Only trained models are involved: the frozen writer proposes, the frozen
reader (eyes + decoder) judges. The receiver still sees only pixels.

Per round, for a batch of sentences:
  1. gradient of the symbolic source NLL w.r.t. the one-hot code ranks all
     36 x 81 single-slot substitutions (first order);
  2. the best proposals (plus nearest-centroid alternatives) are scored exactly
     on the symbolic path;
  3. the best `finalists` are rendered and scored through the eyes on fixed
     views (clean + seeded perturbations): J = mean + 0.5 * max of per-token NLL;
  4. a substitution is accepted only if it lowers J.
At the end the searched code is kept only if its clean greedy reading is not
worse (word edit distance to the source) than the original code's.

  .venv/bin/python v4/search.py eval --split dev --n 300          # paired comparison, PNG round trip
  .venv/bin/python v4/search.py eval --split fresh_random --n 1000
"""
from __future__ import annotations

import argparse
import os
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402


def perturb(img, gen):
    """The training channel (tremor, scale, shift, ink noise) with a fixed generator."""
    b, res = img.shape[0], img.shape[-1]
    r = lambda *s: torch.rand(*s, generator=gen, device="cpu").to(img.device)  # noqa: E731
    ang = (r(b) * 2 - 1) * math.radians(4)
    scale = 1 + (r(b) * 2 - 1) * 0.04
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    shift = (r(b, 2) * 2 - 1) * 0.03
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, (b, 1, res, res), align_corners=False)
    out = F.grid_sample(img[:, None], grid, align_corners=False)[:, 0]
    noise = torch.randn(img.shape, generator=gen, device="cpu").to(img.device)
    return (out + 0.1 * noise).clamp(0, 1)


def key_seed(key, view):
    import hashlib
    return int.from_bytes(hashlib.sha256(f"{view}|{key}".encode()).digest()[:8], "little") & (2**62 - 1)


def perturb_keyed(img, keys, view):
    """Same perturbation for the same (sentence, view) whatever the batch: one generator per sample."""
    out = []
    for i, k in enumerate(keys):
        out.append(perturb(img[i:i + 1], torch.Generator().manual_seed(key_seed(k, view))))
    return torch.cat(out)


class Searcher:
    def __init__(self, model, views=3, finalists=4, proposals=16, near=8, rounds=8, patience=2, view_seed=1234):
        self.m = model
        self.views, self.finalists, self.proposals, self.near = views, finalists, proposals, near
        self.rounds, self.patience, self.view_seed = rounds, patience, view_seed
        self.near_k = 16  # nearest centroids per slot eligible as "near" proposals
        self.mode = "exhaustive"
        self.chunk = int(os.environ.get("HEPTAPOD_CHUNK", 768 if model.dev.type == "cuda" else 32))
        self.S, self.K = H.SLOTS, model.books.shape[1]

    # ---- scoring
    def token_nll(self, mem, labels):
        lg = self.m.t5(encoder_outputs=BaseModelOutput(last_hidden_state=mem), labels=labels).logits.float()
        nll = F.cross_entropy(lg.transpose(1, 2), labels, ignore_index=-100, reduction="none").sum(1)
        return nll / (labels != -100).sum(1).clamp_min(1)

    def sym_mem(self, x):  # x: (N, S, K) symbol distribution
        a = self.m.adapter
        return a.out(a.enc(torch.einsum("bsk,skd->bsd", x, self.m.books) + a.pos))

    def marks(self, idx):
        return self.m.fsq_to_mark[idx]

    @torch.no_grad()
    def visual_J(self, idx, labels, keys):
        """idx (N,S) symbols; labels (N,T); keys (N,) sentence ids. Robust score over clean +
        perturbed views; each sentence always gets the same views (batch-invariant)."""
        img = self.m.draw(self.marks(idx))
        views = [img] + [perturb_keyed(img, keys, f"search{self.view_seed}:{v}") for v in range(self.views)]
        ell = torch.stack([self.token_nll(self.m.memory(self.m.eye_probs(v)), labels) for v in views], 1)
        return ell.mean(1) + 0.5 * ell.max(1).values

    @torch.no_grad()
    def greedy_errors(self, idx, texts):
        preds = self.m.generate(self.m.eye_probs(self.m.draw(self.marks(idx))))
        return torch.tensor([H.edit(H.norm(t).split(), H.norm(p).split()) for t, p in zip(texts, preds)])

    # ---- exhaustive symbolic search (measured best), with a visual robustness guard
    @torch.no_grad()
    def exhaustive(self, texts, near_k=16, rounds=20, chunk=None):
        chunk = chunk or self.chunk  # sequences per decoder call: logits are chunk x T x 32k floats
        m, S, K = self.m, self.S, self.K
        raw = m.writer_out(texts).float()
        dist = ((raw[:, :, None, :] - m.books[None]) ** 2).sum(-1)
        alt = dist.topk(near_k, largest=False).indices                          # (B,S,near) nearest symbols, [0] = writer's own
        idx0 = alt[..., 0].clone()
        labels = m.labels(texts)
        B = len(texts)
        idx = idx0.clone()
        best = self.token_nll(self.sym_mem(F.one_hot(idx, K).float()), labels)
        active = torch.ones(B, dtype=torch.bool, device=m.dev)
        accepted = torch.zeros(B, dtype=torch.long, device=m.dev)
        P = S * near_k
        for _ in range(rounds):
            act = active.nonzero().flatten()
            if len(act) == 0:
                break
            cand = idx[act][:, None].repeat(1, P, 1)                           # (A,P,S)
            slot = torch.arange(S, device=m.dev).repeat_interleave(near_k)
            sym = alt[act].reshape(len(act), -1)                                # (A,P) incl. the writer's own symbol
            cand[:, torch.arange(P, device=m.dev), slot] = sym
            flat, lab = cand.view(-1, S), labels[act].repeat_interleave(P, 0)
            sc = torch.cat([self.token_nll(self.sym_mem(F.one_hot(flat[i:i + chunk], K).float()), lab[i:i + chunk])
                            for i in range(0, len(flat), chunk)]).view(len(act), P)
            sc = torch.where((sym == idx[act].gather(1, slot[None].expand(len(act), -1))), torch.full_like(sc, float("inf")), sc)
            v, j = sc.min(1)
            better = v < best[act] - 1e-4
            sel = act[better]
            idx[sel] = cand[better, j[better]]
            best[sel] = v[better]
            accepted[sel] += 1
            active[act[~better]] = False
        # robustness guard: keep the search only if it reads at least as well on clean + perturbed views
        J0, J1 = self.visual_J(idx0, labels, texts), self.visual_J(idx, labels, texts)
        e0, e1 = self.greedy_errors(idx0, texts), self.greedy_errors(idx, texts)
        worse = ((J1 > J0) | (e1 > e0).to(m.dev))
        idx[worse] = idx0[worse]
        return {"idx0": idx0, "idx": idx, "accepted": accepted, "reverted": worse}

    # ---- gradient-shortlisted search
    def __call__(self, texts):
        if self.mode == "exhaustive":
            return self.exhaustive(texts, rounds=self.rounds)
        m, S, K = self.m, self.S, self.K
        dev = m.dev
        with torch.no_grad():
            raw = m.writer_out(texts).float()
            dist = ((raw[:, :, None, :] - m.books[None]) ** 2).sum(-1)            # (B,S,K)
            idx0 = dist.argmin(-1)
            near = torch.zeros_like(dist, dtype=torch.bool)
            near.scatter_(2, dist.topk(min(K, self.near_k + 1), largest=False).indices, True)
        labels = m.labels(texts)
        B = len(texts)
        idx = idx0.clone()
        J = self.visual_J(idx, labels, texts)
        stall = torch.zeros(B, dtype=torch.long, device=dev)
        accepted = torch.zeros(B, dtype=torch.long, device=dev)
        for _ in range(self.rounds):
            active = (stall < self.patience).nonzero().flatten()
            if len(active) == 0:
                break
            ia, la = idx[active], labels[active]
            x = F.one_hot(ia, K).float().requires_grad_(True)
            with torch.enable_grad():
                nll = self.token_nll(self.sym_mem(x), la).sum()
                g, = torch.autograd.grad(nll, x)
            gain = g - (g * x).sum(-1, keepdim=True)                            # first-order change in NLL
            gain = gain.masked_fill(x.bool(), float("inf"))
            flat = gain.view(len(active), -1)
            top = flat.topk(self.proposals, largest=False).indices
            nflat = gain.masked_fill(~near[active], float("inf")).view(len(active), -1)
            ntop = nflat.topk(self.near, largest=False).indices
            props = torch.cat([top, ntop], 1)                                   # (A, P) flat (slot*K+k)
            P = props.shape[1]
            cand = ia[:, None].repeat(1, P, 1)                                  # (A,P,S)
            slot, sym = props // K, props % K
            cand.scatter_(2, slot[..., None], sym[..., None])
            with torch.no_grad():
                sc = self.token_nll(self.sym_mem(F.one_hot(cand.view(-1, S), K).float()), la.repeat_interleave(P, 0))
            sc = sc.view(len(active), P)
            sc = torch.where(torch.isfinite(flat.gather(1, props)), sc, torch.full_like(sc, float("inf")))  # never "substitute" the current symbol
            fin = sc.topk(self.finalists, largest=False).indices                # (A,F)
            fcand = cand.gather(1, fin[..., None].expand(-1, -1, S)).view(-1, S)
            fkeys = [texts[a] for a in active.tolist() for _ in range(self.finalists)]
            fJ = self.visual_J(fcand, la.repeat_interleave(self.finalists, 0), fkeys).view(len(active), self.finalists)
            best, bi = fJ.min(1)
            better = best < J[active] - 1e-4
            sel = active[better]
            idx[sel] = fcand.view(len(active), self.finalists, S)[better, bi[better]]
            J[sel] = best[better]
            accepted[sel] += 1
            stall[active[~better]] += 1
            stall[sel] = 0
        # keep the searched code only if its clean reading is not worse
        e0, e1 = self.greedy_errors(idx0, texts), self.greedy_errors(idx, texts)
        worse = (e1 > e0).to(dev)
        idx[worse] = idx0[worse]
        return {"idx0": idx0, "idx": idx, "accepted": accepted, "reverted": worse}


# ---------------------------------------------------------------- evaluation


def metrics(texts, preds):
    out = H.score(texts, preds)
    return out


@torch.no_grad()
def read_png(model, marks):
    """Export each ring as a 16-bit PNG, read it back from disk, decode with the eyes."""
    imgs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(len(marks)):
            p = Path(tmp) / f"{i}.png"
            H.save_png(p, H.export_image(model, marks[i:i + 1]))
            imgs.append(H.load_png(p))
    img = torch.stack(imgs).to(model.dev)
    return model.generate(model.eye_probs(img)), img


def fresh_random(n, seed=3):
    s = H.Sampler(random_frac=1.0, seed=seed)
    return [s.random_sentence() for _ in range(n)]


def evaluate(args):
    model, _ = H.load(H.OUT / args.ckpt, H.device())
    model.eval()
    if True:
        rows = H.load_rows(args.split)
        texts = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
        if args.panel == "B":  # confirmation panel: disjoint from panel A (seed 0), never used for selection
            a_set = set(random.Random(0).sample(rows, min(args.n, len(rows))))
            rest = [r for r in rows if r not in a_set]
            texts = random.Random(1).sample(rest, min(args.n, len(rest)))
    srch = Searcher(model, rounds=args.rounds, proposals=args.proposals, near=args.near, patience=args.patience)
    srch.mode = args.mode
    res = {"plain": [], "search": [], "plain_pert": [], "search_pert": [], "shuffled": []}
    acc = rev = 0
    t0 = time.time()
    for i in range(0, len(texts), args.batch):
        if args.max_sec and time.time() - t0 > args.max_sec:
            texts = texts[:i]
            print({"event": "time cap reached", "evaluated": i}, flush=True)
            break
        chunk = texts[i:i + args.batch]
        if args.plain_only:
            with torch.no_grad():
                raw = model.writer_out(chunk).float()
                i0 = ((raw[:, :, None, :] - model.books[None]) ** 2).sum(-1).argmin(-1)
            out = {"idx0": i0, "idx": i0, "accepted": torch.zeros(len(chunk)), "reverted": torch.zeros(len(chunk))}
        else:
            out = srch(chunk)
        acc += out["accepted"].float().sum().item()
        rev += out["reverted"].sum().item()
        for name, idx in (("plain", out["idx0"]), ("search", out["idx"])):
            preds, img = read_png(model, model.fsq_to_mark[idx])
            res[name] += preds
            res[name + "_pert"] += model.generate(model.eye_probs(perturb_keyed(img, chunk, "eval")))  # unseen views
            if name == "search":
                res["shuffled"] += model.generate(model.eye_probs(img[torch.randperm(len(img), device=img.device)]))
        print(f"{min(i + args.batch, len(texts))}/{len(texts)}", {k: metrics(texts[:len(v)], v)["exact"] for k, v in res.items()},
              round(time.time() - t0), "s", flush=True)
    report = {"split": args.split, "n": len(texts), "seed": args.seed, "rounds": args.rounds,
              "sec_per_sentence": round((time.time() - t0) / len(texts), 3),
              "mean_accepted_moves": round(acc / len(texts), 2), "reverted": round(rev / len(texts), 4),
              **{k: metrics(texts, v) for k, v in res.items()},
              "examples": [[t, a, b] for t, a, b in zip(texts[:10], res["plain"][:10], res["search"][:10])]}
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1))
    (H.OUT / f"preds_{Path(args.ckpt).stem}_{args.split}{'_B' if args.panel == 'B' else ''}_{len(texts)}.json").write_text(
        json.dumps({"texts": texts, **{k: v for k, v in res.items()}}))
    (H.OUT / f"search_eval_{Path(args.ckpt).stem}_{args.split}{'_B' if args.panel == 'B' else ''}_{args.n}.json").write_text(json.dumps(report, indent=1))


def main():
    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=True)  # a stalled run shows where it is stuck
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--split", default="dev", choices=["dev", "dev_random", "dev_uniform10", "fresh_natural", "fresh_random",
                                                       "locked_natural", "locked_random", "locked_uniform10"])
    e.add_argument("--n", type=int, default=300)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--rounds", type=int, default=20)
    e.add_argument("--mode", default="exhaustive", choices=["exhaustive", "gradient"])
    e.add_argument("--proposals", type=int, default=16)
    e.add_argument("--near", type=int, default=8)
    e.add_argument("--patience", type=int, default=2)
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--ckpt", default="model.pt")
    e.add_argument("--panel", default="A", choices=["A", "B"])
    e.add_argument("--plain-only", action="store_true", help="no search: the writer as is")
    e.add_argument("--max-sec", type=float, default=0, help="stop and report on what was evaluated (0 = no cap)")
    args = ap.parse_args()
    {"eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
