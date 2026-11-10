#!/usr/bin/env python3
"""Checks on a trained v4 checkpoint, run locally (no GPU needed).

  verify   Read and verify: the reader proposes `--beam` candidate sentences from
           the image; the frozen writer redraws each candidate, and a candidate
           whose drawing matches the received ring (in pixels) wins. Inputs are
           only the image and the reader's own candidates.
  png      Export rings as high-resolution PNGs (as `say` does), read the files
           back (as `read` does) and compare with the in-memory reading.
  final    The once-only test run: full report (visual, symbolic, shuffled-image
           control) plus read-and-verify, on test and test_random.

  HEPTAPOD_OUT=<run dir> .venv/bin/python v4/check_v4.py verify --split dev --beam 32
  HEPTAPOD_OUT=<run dir> .venv/bin/python v4/check_v4.py png --n 200
  HEPTAPOD_OUT=<run dir> .venv/bin/python v4/check_v4.py final
"""
import argparse
import collections
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402

HI = False  # export the training-resolution ring (upsampled) instead of the separate high-res renderer


@torch.no_grad()
def verify(model, split, beam, n=None, batch=16, tol=1e-4):
    texts = H.load_rows(split)[:n]
    stats = collections.Counter()
    words_ok = words = 0
    t0 = time.time()
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        img = model.draw(model.code(chunk))                       # the received rings
        probs = model.eye_probs(img)
        greedy = model.generate(probs)
        cands = model.generate(probs, beams=beam)
        cands = [cands[b * beam:(b + 1) * beam] for b in range(len(chunk))]
        pool = sorted({c for cs in cands for c in cs} | set(greedy))
        redraw = {}
        for j in range(0, len(pool), 128):
            part = pool[j:j + 128]
            for c, im in zip(part, model.draw(model.code(part))):
                redraw[c] = im
        for b, gold in enumerate(chunk):
            options = list(dict.fromkeys(cands[b] + [greedy[b]]))
            dist = {c: (redraw[c] - img[b]).abs().max().item() for c in options}
            match = [c for c in options if dist[c] < tol]
            chosen = match[0] if match else greedy[b]
            g = H.norm(gold)
            stats["greedy"] += H.norm(greedy[b]) == g
            stats["beam_top1"] += H.norm(cands[b][0]) == g
            stats["gold_in_candidates"] += any(H.norm(c) == g for c in options)
            stats["verified"] += H.norm(chosen) == g
            stats["any_match"] += bool(match)
            stats["multiple_matches"] += len(match) > 1
            stats["false_match"] += bool(match) and H.norm(chosen) != g
            gw, pw = g.split(), H.norm(chosen).split()
            words_ok += sum(a == c for a, c in zip(gw, pw))
            words += len(gw)
        done = min(i + batch, len(texts))
        if (i // batch) % 10 == 0:
            print({k: round(v / done, 4) for k, v in stats.items()}, f"{done}/{len(texts)}", round(time.time() - t0), "s", flush=True)
    out = {"split": split, "n": len(texts), "beam": beam, **{k: round(v / len(texts), 4) for k, v in stats.items()},
           "verified_word_acc": round(words_ok / words, 4)}
    print(json.dumps(out, indent=1))
    return out


@torch.no_grad()
def png(model, n):
    import matplotlib.pyplot as plt
    from PIL import Image
    texts = H.load_rows("dev")[:n // 2] + H.load_rows("dev_random")[:n - n // 2]
    hi = H.hi_script().to(model.dev) if HI else None
    same = gold_mem = gold_png = 0
    examples = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(texts):
            code = model.code([t])
            mem = model.generate(model.eye_probs(model.draw(code)))[0]
            img = H.export_image(model, code, hi)
            path = Path(tmp) / f"{i}.png"
            H.save_png(path, img)
            back = model.generate(model.eye_probs(H.load_png(path).to(model.dev)[None]))[0]
            same += mem == back
            gold_mem += H.norm(mem) == H.norm(t)
            gold_png += H.norm(back) == H.norm(t)
            if len(examples) < 6:
                examples.append([t, mem, back])
    out = {"n": len(texts), "png_equals_memory": round(same / len(texts), 4),
           "exact_memory": round(gold_mem / len(texts), 4), "exact_png": round(gold_png / len(texts), 4),
           "examples": examples}
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["verify", "png", "final"])
    ap.add_argument("--ckpt", default="model.pt")
    ap.add_argument("--split", default="dev", choices=["dev", "dev_random", "test", "test_random"])
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--n", type=int)
    args = ap.parse_args()
    model, _ = H.load(H.OUT / args.ckpt, H.device())
    model.eval()
    if args.what == "verify":
        res = verify(model, args.split, args.beam, args.n)
        (H.OUT / f"verify_{args.split}.json").write_text(json.dumps(res, indent=1))
    elif args.what == "png":
        res = png(model, args.n or 200)
        (H.OUT / "png_check.json").write_text(json.dumps(res, indent=1))
    else:
        marker = H.OUT / "TEST_RUN_DONE"
        if marker.exists():
            sys.exit(f"test already run once ({marker}); refusing to run it again")
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        rep = H.report(model, "test")
        ver = {s: verify(model, s, args.beam) for s in ("test", "test_random")}
        (H.OUT / "final_test.json").write_text(json.dumps({"report": rep, "verify": ver}, indent=1))


if __name__ == "__main__":
    main()
