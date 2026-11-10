#!/usr/bin/env python3
"""Two follow-up tests (debate 4, round 3).

cliff   Why do sentences with many T5 pieces fail? Dev sentences (mixed_val, never used for
        training or selection), 9-12 words, split into LIGHT (<=12 T5 tokens incl. EOS) and
        DENSE (>=17). For each: symbolic greedy (no image), visual greedy (eyes), teacher-forced
        exactness (every next token is the argmax given the gold prefix), and sender search
        with 20 vs 40 rounds (visual). Reading the outcomes:
          search 40 >> 20 on dense       -> optimisation budget
          teacher-forced ok, greedy fails -> autoregressive instability
          symbolic also fails             -> code/reader limit, not vision
fresh   Fresh readers on 20k plain pairs, same steps and data: SHARED mark table (as before)
        vs SLOT-SPECIFIC tables (slot x mark embeddings, like the original reader's codebooks,
        but freshly initialised). Evaluated on plain and searched rings of 128 dev web sentences.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402
import search as SR  # noqa: E402
import fresh_reader as FR  # noqa: E402


def log(row, f):
    print(row, flush=True)
    f.write(json.dumps(row) + "\n")
    f.flush()


@torch.no_grad()
def teacher_forced_exact(m, probs, texts):
    labels = m.labels(texts)
    mem = m.memory(probs)
    lg = m.t5(encoder_outputs=BaseModelOutput(last_hidden_state=mem), labels=labels).logits
    ok = (lg.argmax(-1) == labels) | (labels == -100)
    return ok.all(1).float().cpu().tolist()


def cliff(m, args, f):
    tok = m.tok
    pool = [s for s in H.load_rows("mixed_val") if 9 <= len(s.split()) <= 12]
    n_tok = {s: len(tok(s).input_ids) for s in pool}
    light = [s for s in pool if n_tok[s] <= 12]
    dense = [s for s in pool if n_tok[s] >= 17]
    rng = random.Random(0)
    groups = {"light": rng.sample(light, min(args.cliff_n, len(light))), "dense": rng.sample(dense, min(args.cliff_n, len(dense)))}
    out = {}
    for name, texts in groups.items():
        res = {"n": len(texts), "mean_tokens": round(sum(n_tok[s] for s in texts) / len(texts), 1)}
        sym = m.code(texts)
        onehot = F.one_hot(sym, m.script.n_marks).float()
        with H.autocast(m.dev):
            res["symbolic_greedy"] = H.score(texts, m.generate(onehot))
            img = m.draw(sym)
            probs = m.eye_probs(img)
            res["visual_greedy"] = H.score(texts, m.generate(probs))
            res["teacher_forced_exact_symbolic"] = round(sum(teacher_forced_exact(m, onehot, texts)) / len(texts), 4)
            res["teacher_forced_exact_visual"] = round(sum(teacher_forced_exact(m, probs, texts)) / len(texts), 4)
        for R in (20, 40):
            srch = SR.Searcher(m, rounds=R)
            idx = torch.cat([srch(texts[i:i + 32])["idx"] for i in range(0, len(texts), 32)])
            with H.autocast(m.dev):
                marks = m.fsq_to_mark[idx]
                res[f"search{R}_visual"] = H.score(texts, m.generate(m.eye_probs(m.draw(marks))))
                res[f"search{R}_teacher_forced_visual"] = round(sum(teacher_forced_exact(m, m.eye_probs(m.draw(marks)), texts)) / len(texts), 4)
        out[name] = res
        log({"test": "cliff", "group": name, **res}, f)
    return out


class SlotReader(FR.FreshReader):
    """Fresh reader whose input embedding is slot-specific (S x marks x D), freshly initialised."""

    def __init__(self, n_marks, dev):
        super().__init__(n_marks, dev)
        self.slot_emb = nn.Parameter(torch.randn(H.SLOTS, n_marks, H.D, device=dev) * 0.02)

    def _mem(self, probs):
        a = self.adapter
        return a.out(a.enc(torch.einsum("bsk,skd->bsd", probs, self.slot_emb) + a.pos))

    def loss(self, probs, labels):
        return self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=self._mem(probs)), labels=labels).loss

    @torch.no_grad()
    def generate(self, probs):
        mem = self._mem(probs)
        return self.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                                attention_mask=torch.ones(mem.shape[:2], dtype=torch.long, device=mem.device),
                                max_new_tokens=H.MAX_T, num_beams=1, do_sample=False)


def fresh(m, args, f):
    n = m.script.n_marks
    pool = H.load_rows("mixed_train")
    random.Random(3).shuffle(pool)
    texts = pool[:args.fresh_size]
    tcodes = FR.codes(m, texts)
    panel = random.Random(0).sample(H.load_rows("fresh_natural"), args.panel_n)
    pcodes = FR.codes(m, panel)
    srch = SR.Searcher(m, rounds=20)
    idx = torch.cat([srch(panel[i:i + 64])["idx"].cpu() for i in range(0, len(panel), 64)])
    scodes = m.fsq_to_mark.cpu()[idx]
    out = {}
    for name, cls in (("shared_table", FR.FreshReader), ("slot_tables", SlotReader)):
        torch.manual_seed(0)
        reader = cls(n, m.dev)
        params = [p for p in reader.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)  # same optimiser for both readers
        t0 = time.time()
        reader.train()
        gen = torch.Generator().manual_seed(1)
        for step in range(1, args.fresh_steps + 1):
            pick = torch.randint(0, len(texts), (64,), generator=gen)
            probs = F.one_hot(tcodes[pick].to(m.dev), n).float()
            with H.autocast(m.dev):
                loss = reader.loss(probs, m.labels([texts[i] for i in pick.tolist()]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 1000 == 0:
                log({"test": "fresh", "reader": name, "step": step, "loss": round(loss.item(), 4), "sec": round(time.time() - t0)}, f)
        reader.eval()
        res = {"plain_rings": H.score(panel, FR.read_panel(m, reader, pcodes)),
               "searched_rings": H.score(panel, FR.read_panel(m, reader, scodes))}
        out[name] = res
        log({"test": "fresh", "reader": name, **res}, f)
        del reader, opt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--cliff-n", type=int, default=64)
    ap.add_argument("--fresh-size", type=int, default=20000)
    ap.add_argument("--fresh-steps", type=int, default=6000)
    ap.add_argument("--panel-n", type=int, default=128)
    ap.add_argument("--only", choices=["cliff", "fresh"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.cliff_n, args.fresh_size, args.fresh_steps, args.panel_n = 3, 32, 2, 4
    H.seed_all(0)
    m, _ = H.load(Path(args.init), H.device())
    m.eval()
    f = open(H.OUT / "followup_log.jsonl", "a")
    res = {}
    if args.only != "fresh":
        res["cliff"] = cliff(m, args, f)
    if args.only != "cliff":
        res["fresh"] = fresh(m, args, f)
    (H.OUT / "followup.json").write_text(json.dumps(res, indent=1))
    log({"event": "all done"}, f)


if __name__ == "__main__":
    main()
