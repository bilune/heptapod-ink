#!/usr/bin/env python3
"""Non-learned reference points for the ring (evaluation only; never part of the system).

1. channel   How well do the trained eyes read ARBITRARY codewords? Uniform random
             symbols from the 81 marks, rendered, exported as 16-bit PNG blocks
             (exact), read by the eyes; also with keyed unseen perturbations.
             Symbol error rate, errors-per-ring histogram, and the block success
             a Reed-Solomon code over GF(81) would get: RS(36,32) corrects 2
             symbol errors (202 payload bits), RS(36,28) corrects 4 (177 bits).
             Same statistics on the writer's own codes for comparison.
2. source    How many bits does each sentence need?
             - random combinations: exact-distribution code (length + word ids under
               the sampler's own weights), plus uniform 10-word combinations;
             - natural text: ideal arithmetic-coding length under a causal LM, per
               sentence, EOS included, +2 bits coder overhead.
             Fit fractions for 228 (raw), 202 and 177 (RS) payload bits.
3. combined  Classical exact rate = P(fits) x P(RS decodes) per set.

  .venv/bin/python v4/references.py --lm gpt2 --lm Qwen/Qwen2.5-1.5B
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402
import search as SR  # noqa: E402

BUDGETS = {"raw_228": 36 * math.log2(81), "rs36_32_202": 32 * math.log2(81), "rs36_28_177": 28 * math.log2(81)}


@torch.no_grad()
def channel(model, n_rings=2048, batch=64):
    K = len(model.fsq_to_mark)
    out = {}
    g = torch.Generator().manual_seed(11)
    sources = {
        "uniform_codewords": lambda i: torch.randint(0, K, (batch, H.SLOTS), generator=g),
        "writer_natural": lambda i: None,
        "writer_random": lambda i: None,
    }
    nat = H.load_rows("fresh_natural")
    rnd = H.load_rows("fresh_random")
    for name in sources:
        errs = {"clean": [], "perturbed": []}
        for i in range(0, n_rings, batch):
            if name == "uniform_codewords":
                sym = sources[name](i).to(model.dev)
                keys = [f"u{i + j}" for j in range(batch)]
            else:
                if i >= 1000:
                    break
                texts = (nat if name == "writer_natural" else rnd)[i:i + batch]
                raw = model.writer_out(texts).float()
                sym = ((raw[:, :, None, :] - model.books[None]) ** 2).sum(-1).argmin(-1)
                keys = texts
            marks = model.fsq_to_mark[sym]
            imgs = []
            for j in range(len(marks)):  # 16-bit PNG path, exact by construction: reproduce it in memory
                big = torch.tensor(H.export_image(model, marks[j:j + 1]))
                q = torch.round((1 - big) * 65535) / 65535
                imgs.append(F.avg_pool2d((1 - q)[None, None], big.shape[-1] // H.RES)[0, 0])
            img = torch.stack(imgs).to(model.dev)
            for view, im in (("clean", img), ("perturbed", SR.perturb_keyed(img, keys, "refs"))):
                read = model.eye_probs(im).argmax(-1)
                errs[view].append((read != marks).sum(-1).cpu())
        res = {}
        for view, e in errs.items():
            e = torch.cat(e).numpy()
            res[view] = {"rings": int(len(e)), "symbol_error_rate": round(float(e.mean() / H.SLOTS), 5),
                         "all_correct": round(float((e == 0).mean()), 4),
                         "rs36_32_decodes(<=2 errors)": round(float((e <= 2).mean()), 4),
                         "rs36_28_decodes(<=4 errors)": round(float((e <= 4).mean()), 4),
                         "errors_hist_0to6": [int((e == k).sum()) for k in range(7)]}
        out[name] = res
        print(name, json.dumps(res), flush=True)
    return out


def random_bits(texts, vocab):
    w = np.array(vocab["weights"], dtype=np.float64)
    p = dict(zip(vocab["words"], w / w.sum()))
    bits = []
    for t in texts:
        ws = t.split()
        if not all(x in p for x in ws):
            bits.append(float("inf"))
            continue
        bits.append(math.log2(8) + sum(-math.log2(p[x]) for x in ws))
    return np.array(bits)


@torch.no_grad()
def lm_bits(texts, name, dev, batch=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32 if dev.type != "cuda" else torch.bfloat16).to(dev).eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    eos = tok.eos_token_id
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ids = [[bos] + tok(t, add_special_tokens=False).input_ids + [eos] for t in chunk]
        T = max(map(len, ids))
        x = torch.full((len(ids), T), eos, dtype=torch.long)
        mask = torch.zeros(len(ids), T, dtype=torch.bool)
        for j, s in enumerate(ids):
            x[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = True
        x, mask = x.to(dev), mask.to(dev)
        logp = F.log_softmax(lm(x, attention_mask=mask.long()).logits.float(), -1)
        tgt = x[:, 1:]
        lp = logp[:, :-1].gather(-1, tgt[..., None])[..., 0] * mask[:, 1:]
        out += (-(lp.sum(1)) / math.log(2) + 2.0).tolist()          # +2 bits arithmetic-coder overhead
    del lm
    return np.array(out)


def summary(bits):
    fin = bits[np.isfinite(bits)]
    return {"n": int(len(bits)), "median": round(float(np.median(fin)), 1), "p95": round(float(np.percentile(fin, 95)), 1),
            "p99": round(float(np.percentile(fin, 99)), 1), "max": round(float(fin.max()), 1),
            **{f"fits_{k}": round(float((bits <= v).mean()), 4) for k, v in BUDGETS.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", action="append", default=[])
    ap.add_argument("--rings", type=int, default=2048)
    args = ap.parse_args()
    dev = H.device()
    model, _ = H.load(H.OUT / "model.pt", dev)
    model.eval()
    report = {"channel": channel(model, args.rings)}
    vocab = json.loads((H.DATA / "random_vocab.json").read_text())
    rng = random.Random(5)
    w = vocab["weights"]
    uniform10 = [" ".join(rng.choice(vocab["words"]) for _ in range(10)) for _ in range(2000)]
    weighted10 = [" ".join(rng.choices(vocab["words"], weights=w, k=10)) for _ in range(2000)]
    src = {"fresh_random": random_bits(H.load_rows("fresh_random"), vocab),
           "dev_random": random_bits(H.load_rows("dev_random"), vocab),
           "weighted_10_words": random_bits(weighted10, vocab)}
    p_uni = math.log2(len(vocab["words"]))
    src["uniform_10_words(uniform code)"] = np.full(len(uniform10), 10 * p_uni)
    report["source_random"] = {k: summary(v) for k, v in src.items()}
    print(json.dumps(report["source_random"], indent=1), flush=True)
    report["source_natural"] = {}
    for name in args.lm:
        for split in ("dev", "fresh_natural"):
            texts = H.load_rows(split)
            b = lm_bits(texts, name, dev)
            report["source_natural"][f"{name}|{split}"] = {**summary(b), "bits_per_char_median": round(float(np.median(b / np.array([len(t) for t in texts]))), 3)}
            print(name, split, json.dumps(report["source_natural"][f"{name}|{split}"]), flush=True)
    ch = report["channel"]["uniform_codewords"]
    comb = {}
    for k, s in list(report["source_random"].items()) + list(report["source_natural"].items()):
        for view in ("clean", "perturbed"):
            comb[f"{k}|{view}"] = {"rs36_32": round(s["fits_rs36_32_202"] * ch[view]["rs36_32_decodes(<=2 errors)"], 4),
                                   "rs36_28": round(s["fits_rs36_28_177"] * ch[view]["rs36_28_decodes(<=4 errors)"], 4)}
    report["classical_exact_estimate"] = comb
    (H.OUT / "references.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(comb, indent=1))


if __name__ == "__main__":
    main()
