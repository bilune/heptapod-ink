#!/usr/bin/env python3
"""Is it a language? A fresh reader learns the frozen code from N paired examples.

The sender (T5 encoder, slot writer, codebooks, marks, eyes) is frozen. The fresh
reader gets NOTHING from the reader that co-invented the code: new mark embeddings,
a new adapter, and base pretrained google-t5/t5-small (not our fine-tuned decoder).
Training pairs: the sender's plain code for N sentences from the mixed training pool
(clean symbols, as the original reader was trained). Evaluation: held-out panels read
from the rendered ring by the frozen eyes (their mark probabilities), greedy decoding.
Also: the same fresh readers on rings produced by sender-side search (search was done
against the ORIGINAL reader, so this measures whether the better code transfers).

  HEPTAPOD_OUT=<dir> python v4/fresh_reader.py --init model.pt --sizes 1000 5000 20000 200000
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
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402
import search as SR  # noqa: E402


class FreshReader(torch.nn.Module):
    def __init__(self, n_marks, dev):
        super().__init__()
        self.adapter = H.Adapter(n_marks)                       # new mark embeddings + adapter
        attn = {"attn_implementation": "eager"} if dev.type == "mps" else {}
        self.t5 = T5ForConditionalGeneration.from_pretrained(H.T5, **attn)   # base pretrained, not ours
        for p in self.t5.encoder.parameters():
            p.requires_grad_(False)
        self.to(dev)

    def loss(self, probs, labels):
        mem = self.adapter(probs)
        return self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=mem), labels=labels).loss

    @torch.no_grad()
    def generate(self, probs):
        mem = self.adapter(probs)
        out = self.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                               attention_mask=torch.ones(mem.shape[:2], dtype=torch.long, device=mem.device),
                               max_new_tokens=H.MAX_T, num_beams=1, do_sample=False)
        return out


@torch.no_grad()
def codes(model, texts, bs=256):
    return torch.cat([model.code(texts[i:i + bs]).cpu() for i in range(0, len(texts), bs)])


def read_panel(model, reader, marks, bs=128):
    preds = []
    for i in range(0, len(marks), bs):
        m = marks[i:i + bs].to(model.dev)
        with torch.no_grad(), H.autocast(model.dev):
            probs = model.eye_probs(model.draw(m))
            out = reader.generate(probs)
        preds += model.tok.batch_decode(out, skip_special_tokens=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="model.pt")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000, 200000])
    ap.add_argument("--steps", type=int, nargs="+", default=[1500, 3000, 6000, 12000])
    ap.add_argument("--max-sec", type=float, nargs="+", default=[180, 300, 480, 720])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--panel-n", type=int, default=256)
    ap.add_argument("--searched", action="store_true", help="also read sender-searched rings")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.sizes, args.steps, args.max_sec, args.panel_n, args.batch = [16], [3], [30], 8, 4
    H.seed_all(0)
    dev = H.device()
    model, _ = H.load(H.OUT / args.init, dev)
    model.eval()
    f = open(H.OUT / "fresh_reader_log.jsonl", "a")

    def log(row):
        print(row, flush=True)
        f.write(json.dumps(row) + "\n")
        f.flush()

    pool = H.load_rows("mixed_train")
    random.Random(3).shuffle(pool)
    panels = {"web": random.Random(0).sample(H.load_rows("fresh_natural"), args.panel_n),
              "tatoeba": random.Random(0).sample(H.load_rows("dev"), args.panel_n)}
    panel_codes = {k: codes(model, v) for k, v in panels.items()}
    searched = None
    if args.searched:
        srch = SR.Searcher(model, rounds=20)
        idx = torch.cat([srch(panels["web"][i:i + 64])["idx"].cpu() for i in range(0, len(panels["web"]), 64)])
        searched = model.fsq_to_mark.cpu()[idx]
        log({"event": "searched web panel codes ready"})
    # reference: the original reader on the same panels
    ref = {}
    for k, v in panels.items():
        with torch.no_grad(), H.autocast(dev):
            probs = torch.cat([model.eye_probs(model.draw(panel_codes[k][i:i + 128].to(dev))) for i in range(0, len(v), 128)])
            p = []
            for i in range(0, len(v), 128):
                p += model.generate(probs[i:i + 128])
        ref[k] = H.score(v, p)
    log({"event": "original reader", **ref})
    n = model.script.n_marks
    for size, steps, cap in zip(args.sizes, args.steps, args.max_sec):
        texts = pool[:size]
        tcodes = codes(model, texts)
        reader = FreshReader(n, dev)
        opt = torch.optim.AdamW([{"params": reader.adapter.parameters(), "lr": 3e-4},
                                 {"params": [p for n_, p in reader.t5.named_parameters() if not n_.startswith("encoder.")], "lr": 1e-4}],
                                weight_decay=0.01)
        t0, step = time.time(), 0
        reader.train()
        while step < steps and time.time() - t0 < cap:
            step += 1
            pick = torch.randint(0, size, (args.batch,))
            batch = [texts[i] for i in pick.tolist()]
            probs = F.one_hot(tcodes[pick].to(dev), n).float()
            with H.autocast(dev):
                loss = reader.loss(probs, model.labels(batch))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            opt.step()
            if step % 250 == 0:
                log({"size": size, "step": step, "loss": round(loss.item(), 4), "sec": round(time.time() - t0)})
        reader.eval()
        row = {"size": size, "steps_done": step, "train_sec": round(time.time() - t0)}
        for k, v in panels.items():
            row[k] = H.score(v, read_panel(model, reader, panel_codes[k]))
        if searched is not None:
            row["web_searched_rings"] = H.score(panels["web"], read_panel(model, reader, searched))
        log(row)
        del reader, opt
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    log({"event": "all done"})


if __name__ == "__main__":
    main()
