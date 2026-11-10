#!/usr/bin/env python3
"""Regenerate the v4 figures that are not produced by any probe script (English captions).

  .venv/bin/python v4/figures.py hero              # v4/runs/s2/my_sentences_rings.png
  .venv/bin/python v4/figures.py hola              # v4/runs/s2/hola_terricola.png
  .venv/bin/python v4/figures.py analogy           # v4/runs/s2/analogy_queen.png + analogy_woman.gif
  .venv/bin/python v4/figures.py zones             # v4/runs/s2/anatomy_zones.png
  .venv/bin/python v4/figures.py showcase          # v4/runs/final/showcase.png
  .venv/bin/python v4/figures.py showcase_reread   # v4/runs/final/showcase_reread.png
  .venv/bin/python v4/figures.py all

Rings marked "reread" are written with the sender-side search (v4/search.py, 20 rounds).
Every reading goes through the eyes: rings are exported as 16-bit PNGs and read back from disk,
except in the analogy figures, which read the rendered image directly like v4/analogy_probe.py.
On a Mac, keep HEPTAPOD_CHUNK small (32, the default off CUDA): the search is memory hungry.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heptapod_t5 as H  # noqa: E402
import search as SR  # noqa: E402
import analogy_probe as A  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge, Patch  # noqa: E402

S2 = Path("v4/runs/s2")
FINAL = Path("v4/runs/final")
RED = "#c0392b"
ORANGE, BLUE, GREEN = "#d2491f", "#2f6fb3", "#2e8b57"
SEARCH_BATCH = 4  # sentences per search call (keeps the decoder chunks small)


# ---------------------------------------------------------------- helpers


def load(path):
    m, _ = H.load(Path(path), H.device())
    m.eval()
    return m


def slot_angle(k):
    """Centre of slot k in degrees (math convention): slot 0 just above 9 o'clock, clockwise."""
    return 180.0 - (k + 0.5) * 360.0 / H.SLOTS


def plain_idx(m, texts):
    with torch.no_grad():
        raw = m.writer_out(texts).float()
        return ((raw[:, :, None, :] - m.books[None]) ** 2).sum(-1).argmin(-1)


def searched(m, texts):
    """(plain symbols, reread symbols) for each sentence."""
    srch = SR.Searcher(m, rounds=20)
    i0, i1 = [], []
    for i in range(0, len(texts), SEARCH_BATCH):
        out = srch(texts[i:i + SEARCH_BATCH])
        i0.append(out["idx0"])
        i1.append(out["idx"])
        print(f"searched {min(i + SEARCH_BATCH, len(texts))}/{len(texts)}", flush=True)
    return torch.cat(i0), torch.cat(i1)


def read_back(m, idx):
    """Readings through the eyes after a 16-bit PNG round trip, plus images for display."""
    marks = m.fsq_to_mark[idx]
    preds = []
    for i in range(0, len(marks), 8):
        preds += SR.read_png(m, marks[i:i + 8])[0]
    imgs = [H.export_image(m, marks[i:i + 1]) for i in range(len(marks))]
    return preds, imgs


def ok(text, pred):
    return H.norm(text) == H.norm(pred)


def show_ring(ax, img, pad=0):
    r = H.RES
    ax.imshow(1 - img, cmap="gray", vmin=0, vmax=1, extent=(0, r, r, 0), interpolation="antialiased")
    ax.set_xlim(-pad, r + pad)
    ax.set_ylim(r + pad, -pad)
    ax.set_axis_off()


def dots(ax, slots, color, radius=150, size=11):
    c = H.RES / 2
    for s in slots:
        a = math.radians(slot_angle(s))
        ax.plot(c + radius * math.cos(a), c - radius * math.sin(a), "o", color=color, ms=size, clip_on=False)


def wrap(s, width):
    return "\n".join(textwrap.wrap(s, width)) or s


def q(s):
    return f"“{s}”"


# ---------------------------------------------------------------- hero: 20 sentences


def hero():
    m = load(S2 / "model.pt")
    rows = json.loads((S2 / "my_sentences.json").read_text())
    texts = [r["text"] for r in rows]
    i0, i1 = searched(m, texts)
    preds, imgs = read_back(m, i1)
    plain, _ = read_back(m, i0)
    fig, axes = plt.subplots(5, 4, figsize=(16, 22), facecolor="white")
    changed = []
    for ax, t, p, img, r, p0 in zip(axes.flat, texts, preds, imgs, rows, plain):
        show_ring(ax, img)
        good = ok(t, p)
        cap = wrap(t, 38) + "\n" + ("✓ reads back exactly" if good else "✗ reads: " + wrap(p, 46))
        ax.set_title(cap, fontsize=9.5, color="black" if good else RED)
        if p != r["search"] or p0 != r["plain"]:
            changed.append({"text": t, "json_plain": r["plain"], "now_plain": p0, "json_search": r["search"], "now_search": p})
    fig.tight_layout()
    out = S2 / "my_sentences_rings.png"
    fig.savefig(out, dpi=70, facecolor="white")
    print(out, f"exact {sum(ok(t, p) for t, p in zip(texts, preds))}/{len(texts)}")
    print("readings that differ from my_sentences.json:", json.dumps(changed, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- hola, terricola


def hola():
    m = load(S2 / "model.pt")
    texts = ["Hola, terrícola.", "Hello, earthling."]
    _, i1 = searched(m, texts)
    preds, imgs = read_back(m, i1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), facecolor="white")
    for ax, t, p, img in zip(axes, texts, preds, imgs):
        show_ring(ax, img)
        ax.set_title(f"{q(t)}\nthe reader reads: {q(p)}", fontsize=12)
        print(t, "->", p)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = S2 / "hola_terricola.png"
    fig.savefig(out, dpi=90, facecolor="white")
    print(out)


# ---------------------------------------------------------------- analogy: queen - king + man


@torch.no_grad()
def analogy_data(m):
    words = ["king", "queen", "man", "woman"]
    R = dict(zip(words, A.raw(m, words)))
    Q = dict(zip(words, A.quantize(m, torch.stack([R[w] for w in words]))))
    Q["result"] = A.quantize(m, (R["queen"] - R["king"] + R["man"])[None])[0]
    imgs = {k: m.draw(m.fsq_to_mark[v][None])[0] for k, v in Q.items()}
    reading = m.generate(m.eye_probs(imgs["result"][None]))[0]
    hi = {k: H.export_image(m, m.fsq_to_mark[v][None]) for k, v in Q.items()}
    diff = lambda a, b: [s for s in range(H.SLOTS) if int(Q[a][s]) != int(Q[b][s])]  # noqa: E731
    return {"hi": hi, "reading": reading, "kq": diff("king", "queen"), "mr": diff("man", "result"),
            "rw": diff("result", "woman")}


def analogy():
    m = load(S2 / "model.pt")
    d = analogy_data(m)
    kq, mr, rw = d["kq"], d["mr"], d["rw"]
    both = sorted(set(kq) & set(mr))
    print({"king->queen": kq, "man->result": mr, "result_vs_woman": rw, "both": both, "reading": d["reading"]})

    # static figure
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.6), facecolor="white")
    panels = [("king", "The answer was king.", kq, ORANGE), ("queen", "The answer was queen.", kq, ORANGE),
              ("man", "The answer was man.", mr, ORANGE),
              ("result", f"queen − king + man\nreads: {q(d['reading'])}", mr, ORANGE),
              ("woman", "The answer was woman.", [], ORANGE)]
    for ax, (k, title, sl, col) in zip(axes, panels):
        show_ring(ax, d["hi"][k], pad=12)
        dots(ax, sl, col, radius=150, size=12)
        ax.set_title(title, fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.13, wspace=0.12)
    fig.text(0.25, 0.04, f"king → queen: {len(kq)} of 36 marks change (orange dots)", ha="center", fontsize=11)
    fig.text(0.62, 0.04, f"man → result: {len(mr)} of 36 change · result vs 'woman': {len(rw)} marks differ",
             ha="center", fontsize=11)
    fig.text(0.5, 0.005, f"slots that change in both cases: {both}", ha="center", fontsize=9.5, color="#555")
    out = S2 / "analogy_queen.png"
    fig.savefig(out, dpi=80, facecolor="white")
    plt.close(fig)
    print(out)

    # animated GIF, 540 x 630, keyframes with crossfades
    from PIL import Image

    def frame(k, title, sub, sl, col):
        f = plt.figure(figsize=(6.75, 7.875), dpi=80, facecolor="white")
        ax = f.add_axes((0.04, 0.02, 0.92, 0.80))
        show_ring(ax, d["hi"][k], pad=30)
        dots(ax, sl, col, radius=150, size=11)
        f.text(0.5, 0.945, title, ha="center", va="center", fontsize=17, weight="bold")
        f.text(0.5, 0.888, sub, ha="center", va="center", fontsize=11, color="#555")
        f.canvas.draw()
        im = Image.fromarray(np.asarray(f.canvas.buffer_rgba())[..., :3].copy())
        plt.close(f)
        return im

    fem = "“female”"
    king = frame("king", "king", q("The answer was king."), [], ORANGE)
    king_hl = frame("king", "king", f"these {len(kq)} marks are about to change", kq, ORANGE)
    queen_hl = frame("queen", "queen", f"that difference is the idea of {fem}", kq, ORANGE)
    man = frame("man", "man", q("The answer was man."), [], BLUE)
    man_hl = frame("man", "man + (queen − king)", f"we add that difference: only {len(mr)} marks change", mr, BLUE)
    res_hl = frame("result", "the reader looks at this ring…", f"nobody wrote {q('woman')} to it", mr, BLUE)
    res_ok = frame("result", f"and reads: {q(d['reading'])}", f"the idea of {fem} moved to another word, in ink", mr, GREEN)
    # crossfade frames carry the "transition" captions
    kq_a = frame("king", "king → queen", f"{len(kq)} of 36 marks change", kq, ORANGE)
    kq_b = frame("queen", "king → queen", f"{len(kq)} of 36 marks change", kq, ORANGE)
    mr_a = frame("man", "man + (queen − king)", f"{len(mr)} marks change", mr, BLUE)
    mr_b = frame("result", "man + (queen − king)", f"{len(mr)} marks change", mr, BLUE)
    frames, durs = [king, king_hl], [1600, 1400]
    for i in range(10):
        frames.append(Image.blend(kq_a, kq_b, (i + 1) / 11)); durs.append(90)
    frames += [queen_hl, man, man_hl]; durs += [2000, 1600, 1600]
    for i in range(10):
        frames.append(Image.blend(mr_a, mr_b, (i + 1) / 11)); durs.append(90)
    frames += [res_hl, res_ok]; durs += [1800, 3500]
    out = S2 / "analogy_woman.gif"
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)
    print(out, len(frames), "frames")


# ---------------------------------------------------------------- ring zones


def zone_slots():
    """Zones from the probe outputs (zones_causal, content_probe, anatomy)."""
    zc = json.loads((S2 / "zones_causal.json").read_text())["A"]
    cp = json.loads((S2 / "content_probe.json").read_text())["map"]
    an = json.loads((S2 / "anatomy.json").read_text())
    return {
        "who": (zc["who"]["zone"], "who (subject, first word, names)", "#4a86c5"),
        "verb": (zc["verb"]["zone"], "what it does (verb, tense, negation)", "#d9804f"),
        "object": (cp["noun"]["best_slots"][:4], "on what (object noun)", "#c9a227"),
        "end": (zc["end"]["zone"], "how it ends (question, punctuation, names at the end)", "#3f9e6e"),
        "where": (sorted(an["features"]["has_number"]["best_slots"][:3]), "when / where (numbers, time and place words)",
                  "#9b72c4"),
    }, an["n_sentences"]


def zones():
    m = load(S2 / "model.pt")
    text = "Tom didn't go to school yesterday?"
    img = H.export_image(m, m.fsq_to_mark[plain_idx(m, [text])])
    Z, n = zone_slots()
    print({k: v[0] for k, v in Z.items()})
    per_slot = {}
    for k, (sl, _, _) in Z.items():
        for s in sl:
            per_slot.setdefault(s, []).append(k)
    fig = plt.figure(figsize=(9, 10.6), facecolor="white")
    ax = fig.add_axes((0.02, 0.17, 0.96, 0.75))
    show_ring(ax, img, pad=35)
    c, w = H.RES / 2, 360 / H.SLOTS
    for s, ks in per_slot.items():
        a0 = slot_angle(s) - w / 2
        span = w / len(ks)  # a slot in two zones is split in half
        for j, k in enumerate(ks):
            lo, hi = a0 + j * span, a0 + (j + 1) * span
            # the image y axis points down, so screen angle t is data angle -t
            ax.add_patch(Wedge((c, c), 178, -hi, -lo, width=122, facecolor=Z[k][2], alpha=0.35, edgecolor="none"))
    ax.set_title(f"Ring zones (measured over {round(n, -3):,} sentences)\nbackground: the ring for {q(text)}", fontsize=12)
    handles = [Patch(facecolor=col, alpha=0.6, label=f"{lab}  ·  slots {', '.join(map(str, sl))}")
               for sl, lab, col in Z.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=10, frameon=False)
    out = S2 / "anatomy_zones.png"
    fig.savefig(out, dpi=80, facecolor="white")
    print(out)


# ---------------------------------------------------------------- showcases (final checkpoint)


SHOWCASE = ["I left my keys at home.", "Tom put his hand over Mary's hand.",
            "Ziri and Rima learned a very important lesson early in their marriage.",
            "marika my rest what advocate mary", "local hounds here no accordingly",
            "parade anticipated next over theremin loudly turkeys tom stand careers"]
SHOWCASE_REREAD = ["I left my keys at home.", "Lets call the whole thing off.", "Great Service Great People",
                   "differently clippers ensure weakening", "dominate postmodernism homeland rooms beverage sierra"]


def showcase(texts, out, mark_first, row_h):
    m = load(FINAL / "model.pt")
    i0, i1 = searched(m, texts)
    p0, g0 = read_back(m, i0)
    p1, g1 = read_back(m, i1)
    fig = plt.figure(figsize=(9, row_h * len(texts)), facecolor="white")
    subs = fig.subfigures(len(texts), 1)
    for sf, t, a, b, ia, ib in zip(subs, texts, p0, p1, g0, g1):
        sf.suptitle(q(t), fontsize=10.5, weight="bold", y=0.975)
        axes = sf.subplots(1, 2)
        for ax, name, p, img in ((axes[0], "writer as is", a, ia), (axes[1], "writer that rereads", b, ib)):
            show_ring(ax, img)
            mark = "✓" if ok(t, p) else "✗"
            line = f"reads: {mark} {p}" if mark_first else f"reads {mark}: {p}"
            ax.set_title(name + "\n" + wrap(line, 52), fontsize=8)
        sf.subplots_adjust(left=0.02, right=0.98, top=0.79, bottom=0.02, wspace=0.05)
        print(t, "| plain:", a, "| reread:", b)
    fig.savefig(out, dpi=80, facecolor="white")
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figure", choices=["hero", "hola", "analogy", "zones", "showcase", "showcase_reread", "all"])
    a = ap.parse_args()
    jobs = {"hero": hero, "hola": hola, "analogy": analogy, "zones": zones,
            "showcase": lambda: showcase(SHOWCASE, FINAL / "showcase.png", True, 4.6),
            "showcase_reread": lambda: showcase(SHOWCASE_REREAD, FINAL / "showcase_reread.png", False, 4.7)}
    for name in (jobs if a.figure == "all" else [a.figure]):
        jobs[name]()
        plt.close("all")


if __name__ == "__main__":
    main()
