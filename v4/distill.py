#!/usr/bin/env python3
"""Amortize sender-side search into the writer (expert iteration, one or two rounds).

  gen    the frozen v4 system searches better codes for train sentences
         (50% natural Tatoeba train, 50% fresh random combinations);
  train  only the T5 encoder and the slot writer learn: slot cross-entropy toward
         the searched symbols (logits = -distance to the frozen codebooks / tau_s)
         plus text cross-entropy through the hard code and the frozen reader;
         20% of each batch is fresh sentences with text loss only.
  eval   plain writer (no search at send time), read through the eyes, on fixed
         dev panels. Selection: best random-combo word accuracy with natural
         exact no more than 2 points below the starting model.
Eyes, codebooks, adapter and decoder are frozen throughout.

  HEPTAPOD_OUT=<dir> python v4/distill.py --init /runs/v4_vq_ent2/model.pt --gen-sec 1200 --train-sec 1500 --rounds 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402
import search as SR  # noqa: E402


def log(row, f):
    print(row, flush=True)
    f.write(json.dumps(row) + "\n")
    f.flush()


@torch.no_grad()
def panel_eval(model, panels):
    out = {}
    for name, texts in panels.items():
        m, _ = H.evaluate(model, texts, "visual")
        out[name] = {"exact": m["exact"], "words": m["word_acc"], "wer": m["wer"]}
    return out


def generate(model, sampler, args, f, seconds, old=None):
    """Search targets for fresh train sentences until the time cap."""
    model.eval()
    srch = SR.Searcher(model, rounds=args.search_rounds, proposals=args.proposals, near=args.near, patience=args.patience)
    srch.mode = args.search_mode
    texts, idx, Js = [], [], []
    t0 = time.time()
    while time.time() - t0 < seconds:
        half = args.gen_batch // 2
        sampler.random_frac = 0.0
        chunk = sampler.batch(half) + [sampler.random_sentence() for _ in range(args.gen_batch - half)]
        with H.autocast(model.dev):
            out = srch(chunk)
            J = srch.visual_J(out["idx"], model.labels(chunk), chunk)
        texts += chunk
        idx.append(out["idx"].cpu())
        Js.append(J.float().cpu())
        if len(texts) % (args.gen_batch * 20) == 0:
            log({"phase": "gen", "targets": len(texts), "sec": round(time.time() - t0),
                 "moves": round(out["accepted"].float().mean().item(), 2)}, f)
    data = {"texts": texts, "idx": torch.cat(idx), "J": torch.cat(Js)}
    log({"phase": "gen", "event": "done", "targets": len(texts), "per_sec": round(len(texts) / (time.time() - t0), 2)}, f)
    return data


def hinge(model, z, tgt, margin):
    """Decision-region imitation: how far z is on the wrong side of each target half-space
    (in raw-space distance units), max over competitors, hinged at `margin`. (Codex, debate 3 round 3)"""
    B = model.books
    ct = B[torch.arange(H.SLOTS, device=z.device)[None], tgt]
    num = 2 * torch.einsum("bsd,skd->bsk", z, B) - 2 * (z * ct).sum(-1, keepdim=True) \
        + (ct * ct).sum(-1, keepdim=True) - (B * B).sum(-1)[None]
    den = 2 * (ct[:, :, None, :] - B[None]).norm(dim=-1).clamp_min(1e-6)
    viol = (num / den).masked_fill(F.one_hot(tgt, B.shape[1]).bool(), -1e9)
    return torch.relu(margin + viol.max(-1).values)                             # (B, S)


def train(model, sampler, data, tau, args, f, seconds, panels, base):
    for p in model.parameters():
        p.requires_grad_(False)
    enc = list(model.encoder.parameters())
    wri = list(model.writer.parameters())
    for p in enc + wri:
        p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": wri, "lr": args.lr_writer}, {"params": enc, "lr": args.lr_encoder}],
                            weight_decay=0.0 if args.objective == "hinge" else 0.01)
    n = len(data["texts"])
    best, best_row = None, None
    t0, step = time.time(), 0
    while time.time() - t0 < seconds:
        step += 1
        model.eval()                     # no dropout: we imitate deterministic targets
        k = int(args.batch * 0.8)
        pick = torch.randint(0, n, (k,))
        texts = [data["texts"][i] for i in pick.tolist()]
        target = data["idx"][pick].to(model.dev)
        sampler.random_frac = 0.5
        fresh = sampler.batch(args.batch - k)
        if args.objective == "hinge":
            with H.autocast(model.dev):
                raw = model.writer_out(texts).float()
            z0, own = data["z0"][pick].to(model.dev), data["own"][pick].to(model.dev)
            h = hinge(model, raw, target, args.margin)
            ch = own != target
            slot_ce = (h[ch].mean() if ch.any() else 0 * h.sum()) + h[~ch].mean()
            text = 1e-3 * ((raw - z0) ** 2).mean()                               # weak trust region
            loss = slot_ce + text
            d = ((raw[:, :, None, :] - model.books[None]) ** 2).mean(-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc + wri, 1.0)
            opt.step()
        else:
          with H.autocast(model.dev):
            raw = model.writer_out(texts + fresh).float()
            d = ((raw[:, :, None, :] - model.books[None]) ** 2).mean(-1)          # (B,S,K)
            # bounded imitation: pull each slot vector onto its target centroid (minimum is the
            # centroid itself, so the nearest symbol becomes the target); unbounded CE on
            # distances made vectors run away in the first attempt
            goal = model.books[torch.arange(H.SLOTS, device=raw.device)[None], target]
            slot_ce = F.mse_loss(raw[:k], goal)                                   # imitation (same scale as v4's commitment)
            q = model.books[torch.arange(H.SLOTS, device=raw.device)[None], d.argmin(-1)]
            z = raw + (q - raw).detach()
            text = model.text_loss(z, model.labels(texts + fresh))
            commit = F.mse_loss(raw[k:], q[k:].detach())                         # fresh sentences stay near their marks
            loss = slot_ce + 0.25 * commit + args.text_weight * text
          opt.zero_grad(set_to_none=True)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(enc + wri, 1.0)
          opt.step()
        if step % 100 == 0 or step % args.eval_every == 0:
            with torch.no_grad():
                match = (d[:k].argmin(-1) == target).float().mean().item()
            row = {"phase": "train", "step": step, "slot_mse": round(slot_ce.item(), 4), "text": round(text.item(), 4),
                   "raw_norm": round(raw.norm(dim=-1).mean().item(), 2),
                   "target_match": round(match, 4), "sec": round(time.time() - t0)}
            if step % args.eval_every == 0:
                row["panels"] = panel_eval(model, panels)
                ok = row["panels"]["dev"]["exact"] >= base["dev"]["exact"] - 0.02
                score = row["panels"]["dev_random"]["words"]
                if ok and (best is None or score > best):
                    best, best_row = score, row["panels"]
                    H.save(model, H.OUT / "distilled.pt", 2)
            log(row, f)
    return best, best_row


def main():
    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=True)  # a stalled run shows where it is stuck
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--targets", help="reuse searched targets from a previous run for round 1")
    ap.add_argument("--rounds", type=int, default=2, help="expert-iteration rounds (gen + train)")
    ap.add_argument("--gen-sec", type=float, default=1200)
    ap.add_argument("--train-sec", type=float, default=1500)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--search-rounds", type=int, default=20)
    ap.add_argument("--search-mode", default="exhaustive", choices=["exhaustive", "gradient"])
    ap.add_argument("--proposals", type=int, default=64)
    ap.add_argument("--near", type=int, default=64)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-writer", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=0.0)
    ap.add_argument("--text-weight", type=float, default=0.1)
    ap.add_argument("--objective", default="hinge", choices=["hinge", "mse"])
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--panel-n", type=int, default=512)
    ap.add_argument("--min-gain", type=float, default=0.03, help="round-1 stop rule: random-word gain needed to continue")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.gen_sec, args.train_sec, args.gen_batch, args.search_rounds = 20, 20, 4, 2
        args.proposals, args.near, args.batch, args.eval_every, args.panel_n = 8, 8, 8, 1, 8
    H.seed_all(args.seed)
    H.OUT.mkdir(parents=True, exist_ok=True)
    f = open(H.OUT / "distill_log.jsonl", "a")
    model, _ = H.load(Path(args.init), H.device())
    sampler = H.Sampler(seed=args.seed + 100)
    rng = random.Random(7)
    panels = {"dev": rng.sample(H.load_rows("dev"), args.panel_n), "dev_random": rng.sample(H.load_rows("dev_random"), args.panel_n)}
    base = panel_eval(model, panels)
    log({"event": "start", "args": vars(args), "baseline_plain_writer": base}, f)
    tau = H.calibrate_tau(model, sampler).float()
    best_overall = None
    data = None
    for r in range(1, args.rounds + 1):
        if r == 1 and args.targets:
            new = torch.load(args.targets)
            log({"phase": "gen", "event": "loaded", "targets": len(new["texts"]), "from": args.targets}, f)
        else:
            new = generate(model, sampler, args, f, args.gen_sec)
        if data is not None:  # keep the union; old targets stay (they were searched under the same frozen reader)
            new = {"texts": data["texts"] + new["texts"], "idx": torch.cat([data["idx"], new["idx"]]), "J": torch.cat([data["J"], new["J"]])}
        data = new
        torch.save(data, H.OUT / f"targets_round{r}.pt")
        if args.objective == "hinge":
            with torch.no_grad():
                model.eval()
                zs = [model.writer_out(data["texts"][i:i + 256]).float().cpu() for i in range(0, len(data["texts"]), 256)]
            data["z0"] = torch.cat(zs)
            data["own"] = ((data["z0"].to(model.dev)[:, :, None, :] - model.books[None].to(model.dev)) ** 2).sum(-1).argmin(-1).cpu() \
                if len(data["z0"]) < 4096 else torch.cat([((z.to(model.dev)[:, :, None, :] - model.books[None]) ** 2).sum(-1).argmin(-1).cpu() for z in zs])
            log({"event": "init agreement", "value": round((data["own"] == data["idx"]).float().mean().item(), 4)}, f)
        best, row = train(model, sampler, data, tau, args, f, args.train_sec, panels, base)
        log({"event": f"round {r} done", "best_random_words": best, "panels": row}, f)
        if best is None or best < base["dev_random"]["words"] + args.min_gain:
            log({"event": "stop rule: not enough gain", "round": r}, f)
            break
        best_overall = best
        model, _ = H.load(H.OUT / "distilled.pt", model.dev)
    log({"event": "done", "best_random_words": best_overall}, f)


if __name__ == "__main__":
    main()
