#!/usr/bin/env python3
"""Heptapod Ink v2: an emergent script written with the film's own ink.

The 38 reference logograms (Wolfram's Arrival repository, CC BY-NC) are cut
into 912 overlapping "pie slices" of real ink (see ../reference/prep_library.py).
A logogram here is a ring of 12 slots; for each slot the speaker picks one
real fragment. The listener sees only the assembled ring (rotated, noisy,
partly covered) and must recover the whole sentence.

So the marks are the film's, but which marks mean what, and how they combine,
is invented by the two networks.

Usage:
  python heptapod_ink.py train --steps 3000
  python heptapod_ink.py say heptapod offers weapon --tense future
  python heptapod_ink.py poster
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v1"))
import heptapod as H  # noqa: E402  (meanings, listener, channel, metrics)

OUT = HERE / "out"
LIB = np.load(HERE.parent / "reference" / "marks.npz")
R_IN, R_OUT = float(LIB["r_in"]), float(LIB["r_out"])
SLOTS = 12
PITCH_DEG = 360 / SLOTS
RING_DEG = float(LIB["ring_deg"])
MARK_DEG = float(LIB["max_mark_deg"])
RING_R = 0.50                                    # ring radius on the [-1,1] canvas


def with_blank(lib: torch.Tensor) -> torch.Tensor:
    """Index 0 = nothing (a gap in the ring, or no mark)."""
    return torch.cat([torch.zeros_like(lib[:1]), lib])


def ring_rows(lib: torch.Tensor) -> torch.Tensor:
    """Keep only the ring line itself: stray specks in a ring window would be
    cut into hard-edged squares by the window."""
    r = torch.linspace(R_IN, R_OUT, lib.shape[1])
    d = (r - 1).abs()
    # soft radial fade instead of a hard cut, so nothing ends in a square edge
    fade = (0.5 + 0.5 * torch.cos(math.pi * ((d - 0.035) / 0.035).clamp(0, 1)))
    return lib * fade[None, :, None]


def pick_rings(n: int = 32, max_quantile: float = 0.6) -> torch.Tensor:
    """A small, varied set of ring segments, spread from hairline to heavy.

    Ring segments carry little meaning on their own; a huge menu of near-equal
    ones only makes the speaker's choice harder to learn.
    """
    lo = torch.tensor(LIB["rings_lo"])
    r = torch.linspace(R_IN, R_OUT, lo.shape[1])
    # how much ink sits off the thin ring line: heavy segments get clipped
    spill = (lo * ((r - 1).abs() > 0.035).float()[None, :, None]).mean((1, 2))
    order = spill.argsort()[: int(len(spill) * max_quantile)]  # thin and medium only
    weight = lo[order].mean((1, 2))
    order = order[weight.argsort()]
    return order[torch.linspace(0, len(order) - 1, n).round().long()]


RING_PICK = pick_rings()
RINGS_LO = with_blank(ring_rows(torch.tensor(LIB["rings_lo"]))[RING_PICK])
MARKS_LO = with_blank(torch.tensor(LIB["marks_lo"]))
N_RINGS, N_MARKS = len(RINGS_LO), len(MARKS_LO)


def feather(width: int, fade_cols: float) -> torch.Tensor:
    """Angular window: 1 in the middle, cosine fade at both ends."""
    c = torch.arange(width).float() + 0.5
    ramp = (torch.minimum(c, width - c) / max(fade_cols, 1e-6)).clamp(0, 1)
    return 0.5 - 0.5 * torch.cos(math.pi * ramp)


class RingMarkRenderer(nn.Module):
    """A logogram = ring segments + whole marks, one of each per slot.

    Ring segments are feathered into each other (they only hold the ring line,
    so the joins are invisible). Marks are pasted whole, centred on their slot,
    and may spill over neighbouring slots, like merged ink masses in the film.
    """

    def __init__(self, rings: torch.Tensor, marks: torch.Tensor, cpd: float, res: int):
        super().__init__()
        self.cols = int(round(360 * cpd))
        self.rad = rings.shape[1]
        self.rw, self.mw = rings.shape[-1], marks.shape[-1]
        self.register_buffer("rings", rings.reshape(len(rings), -1))
        self.register_buffer("marks", marks.reshape(len(marks), -1))
        self.register_buffer("ring_window", feather(self.rw, (RING_DEG - PITCH_DEG) * cpd))
        centers = [(k + 0.5) * PITCH_DEG * cpd for k in range(SLOTS)]
        self.register_buffer("ring_idx", torch.stack(
            [(torch.arange(self.rw) + round(c - self.rw / 2)) % self.cols for c in centers]))
        self.register_buffer("mark_idx", torch.stack(
            [(torch.arange(self.mw) + round(c - self.mw / 2)) % self.cols for c in centers]))
        lin = torch.linspace(-1, 1, res)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        r_rel = torch.sqrt(x**2 + y**2) / RING_R
        phi = torch.atan2(y, x)
        gy = (r_rel - R_IN) / (R_OUT - R_IN) * 2 - 1
        gx = ((phi + math.pi) / (2 * math.pi) * self.cols + 1) / (self.cols + 2) * 2 - 1
        self.register_buffer("grid", torch.stack([gx, gy], -1)[None])

    def polar(self, ring_choice, mark_choice):
        b = ring_choice.shape[0]
        rings = (ring_choice @ self.rings).view(b, SLOTS, self.rad, self.rw) * self.ring_window
        marks = (mark_choice @ self.marks).view(b, SLOTS, self.rad, self.mw)
        keep = torch.ones(b, self.rad, self.cols, device=ring_choice.device)
        for k in range(SLOTS):
            blank = torch.ones_like(keep)
            blank[..., self.ring_idx[k]] = 1 - rings[:, k].clamp(0, 1)
            keep = keep * blank
            blank = torch.ones_like(keep)
            blank[..., self.mark_idx[k]] = 1 - marks[:, k].clamp(0, 1)
            keep = keep * blank
        return 1 - keep

    def to_image(self, polar):
        p = torch.cat([polar[..., -1:], polar, polar[..., :1]], -1)[:, None]
        return F.grid_sample(p, self.grid.expand(p.shape[0], -1, -1, -1), align_corners=True)[:, 0]

    def forward(self, ring_choice, mark_choice):
        return self.to_image(self.polar(ring_choice, mark_choice))


def st_gumbel(logits: torch.Tensor, tau: float, hard: bool = True) -> torch.Tensor:
    """Straight-through Gumbel-softmax (hard one-hot forward, soft backward).

    Written by hand because torch's gumbel_softmax can emit NaN on MPS.
    """
    u = torch.rand_like(logits).clamp(1e-9, 1 - 1e-9)
    y = F.softmax((logits - torch.log(-torch.log(u))) / tau, dim=-1)
    if not hard:
        return y
    one_hot = F.one_hot(y.argmax(-1), y.shape[-1]).to(y.dtype)
    return one_hot + y - y.detach()


class InkSpeaker(nn.Module):
    def __init__(self, emb: int = 32, hidden: int = 512):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(n, emb) for n in H.SIZES])
        self.net = nn.Sequential(nn.Linear(emb * len(H.SIZES), hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, SLOTS * (N_RINGS + N_MARKS)))
        self.render = RingMarkRenderer(RINGS_LO, MARKS_LO, float(LIB["lo_cpd"]), H.RES)
        self.tau = 1.0
        self.hard = True

    def logits(self, meanings):
        z = torch.cat([e(meanings[:, i]) for i, e in enumerate(self.embeds)], 1)
        out = self.net(z).view(-1, SLOTS, N_RINGS + N_MARKS)
        return out[..., :N_RINGS], out[..., N_RINGS:]

    def choose(self, meanings):
        lr, lm = self.logits(meanings)
        if self.training:
            return st_gumbel(lr, self.tau, self.hard), st_gumbel(lm, self.tau, self.hard)
        return F.one_hot(lr.argmax(-1), N_RINGS).float(), F.one_hot(lm.argmax(-1), N_MARKS).float()

    def forward(self, meanings):
        return self.render(*self.choose(meanings))


# ---------------------------------------------------------------- training


def reference_ink_share() -> float:
    """Ink share of the real logograms drawn in the same 96px ring frame."""
    old = np.load(HERE.parent / "reference" / "fragments.npz")
    strips = torch.tensor(old["strips"]).float() / 255  # (38, rad, 1800) full circles
    rend = RingMarkRenderer(RINGS_LO, MARKS_LO, float(LIB["lo_cpd"]), H.RES)
    p = F.interpolate(strips[:, None], size=(rend.rad, rend.cols), mode="area")[:, 0]
    return float(rend.to_image(p).mean())


def train(args):
    torch.manual_seed(args.seed)
    dev = H.device()
    train_m, held_m = H.split_meanings(args.seed)
    speaker, listener = InkSpeaker().to(dev), H.Listener().to(dev)
    target_ink = reference_ink_share()
    opt = torch.optim.AdamW(list(speaker.parameters()) + list(listener.parameters()), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    print(f"device={dev} rings={N_RINGS - 1} marks={N_MARKS - 1} slots={SLOTS} target_ink={target_ink:.4f}", flush=True)
    history = []
    for step in range(1, args.steps + 1):
        frac = step / args.steps
        speaker.tau = 2.0 * (0.25 ** frac)             # 2.0 -> 0.5
        # Soft start: mixtures of pieces give a clean gradient early on; from
        # 40% of training the speaker must commit to whole pieces.
        speaker.hard = frac >= args.soft_until
        max_angle = math.pi * min(1.0, step / (0.5 * args.steps))
        batch = train_m[torch.randint(len(train_m), (args.batch,))].to(dev)
        img = speaker(batch)
        logits = listener(H.channel(img, max_angle))
        per_slot = torch.stack([F.cross_entropy(l, batch[:, i]) for i, l in enumerate(logits)])
        # Normalise by each slot's chance level, then weight lagging slots up so
        # no word gets abandoned in a bad local optimum.
        chance = torch.log(torch.tensor(H.SIZES, dtype=torch.float32, device=dev))
        rel = per_slot / chance
        ce = (per_slot * (1 + 2 * rel.detach() / rel.detach().mean())).sum() / 3
        ink = img.mean()
        # Expected number of marks per logogram, from the speaker's own odds.
        _, lm = speaker.logits(batch)
        marks = (1 - F.softmax(lm, -1)[..., 0]).sum(-1).mean()
        mark_cost = F.relu(marks - args.marks_target).pow(2) / args.marks_target
        loss = ce + args.ink_weight * (ink - target_ink).pow(2) / target_ink**2 + args.mark_weight * mark_cost
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(), sched.step()
        if step % 250 == 0 or step == 1:
            row = {"step": step, "loss": round(ce.item(), 4), "ink": round(ink.item(), 4), "marks": round(marks.item(), 2),
                   "tau": round(speaker.tau, 3), "rotation_deg": round(math.degrees(max_angle))}
            history.append(row)
            print(row, flush=True)
    OUT.mkdir(exist_ok=True)
    torch.save({"speaker": speaker.state_dict(), "listener": listener.state_dict()}, OUT / "model.pt")
    report = {
        "setup": {"steps": args.steps, "batch": args.batch, "rings": N_RINGS - 1, "marks": N_MARKS - 1, "slots": SLOTS,
                  "target_ink": round(target_ink, 4), "ink_weight": args.ink_weight,
                  "marks_target": args.marks_target, "mark_weight": args.mark_weight},
        "train_meanings_full_rotation": H.evaluate(speaker, listener, train_m, dev),
        "held_out_triples_full_rotation": H.evaluate(speaker, listener, held_m, dev),
    }
    topo, shuf = H.topographic_similarity(speaker, H.all_meanings(), dev)
    report["topographic_similarity"] = {"real": topo, "shuffled_baseline": shuf}
    report["vocabulary"] = vocabulary_stats(speaker, dev)
    report["history"] = history
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    poster(args, speaker=speaker)


@torch.no_grad()
def vocabulary_stats(speaker, dev):
    """Which real ink pieces does the script actually use?"""
    speaker.eval()
    lr, lm = speaker.logits(H.all_meanings().to(dev))
    rc, mc = lr.argmax(-1).cpu(), lm.argmax(-1).cpu()
    return {"distinct_ring_segments": int(len(torch.unique(rc))),
            "distinct_marks": int(len(torch.unique(mc[mc > 0]))),
            "gap_share": round(float((rc == 0).float().mean()), 3),
            "marks_per_logogram": round(float((mc > 0).float().sum(1).mean()), 2)}


# ---------------------------------------------------------------- high-res painting

_HI = {}


def hi_renderer(res: int) -> RingMarkRenderer:
    if res not in _HI:
        rings = with_blank(ring_rows(torch.tensor(LIB["rings_hi"]).float() / 255)[RING_PICK])
        marks = with_blank(torch.tensor(LIB["marks_hi"]).float() / 255)
        _HI[res] = RingMarkRenderer(rings, marks, float(LIB["hi_cpd"]), res)
    return _HI[res]


@torch.no_grad()
def paint(speaker, meanings, dev, res=1024):
    speaker.eval()
    rc, mc = speaker.choose(meanings.to(dev))
    rend = hi_renderer(res)
    out = [rend(r[None].cpu(), m[None].cpu())[0] for r, m in zip(rc, mc)]
    # Ink is ink: crisp edges, no grey halos.
    out = torch.stack(out)
    out = ((out - 0.35) / 0.3).clamp(0, 1)
    return out * out * (3 - 2 * out)


def to_rgb(ink: torch.Tensor) -> np.ndarray:
    a = ink.clamp(0, 1).numpy()
    return np.repeat((1 - a)[..., None], 3, -1)


def load(dev):
    ckpt = torch.load(OUT / "model.pt", map_location=dev)
    speaker, listener = InkSpeaker().to(dev), H.Listener().to(dev)
    speaker.load_state_dict(ckpt["speaker"])
    listener.load_state_dict(ckpt["listener"])
    return speaker.eval(), listener.eval()


SHOWCASE = [(1, 3, 3, 2, 0), (0, 4, 4, 1, 0), (1, 5, 5, 0, 0),
            (0, 2, 1, 1, 1), (2, 1, 0, 2, 0), (5, 4, 4, 2, 0)]


def poster(args, speaker=None):
    dev = H.device()
    if speaker is None:
        speaker, _ = load(dev)
    m = torch.tensor(SHOWCASE)
    imgs = paint(speaker, m, dev, res=640)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.4), facecolor="white")
    for ax, img, mm in zip(axes.flat, imgs, m):
        ax.imshow(to_rgb(img), interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(H.gloss(mm), fontsize=11, color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "poster.png", dpi=110, facecolor="white")
    plt.close(fig)

    # Minimal pairs at high res: change one word at a time.
    base = torch.tensor([[1, 3, 3, 0, 0]])
    rows = [base]
    labels = [H.gloss(base[0]), "subject → human", "verb → understands", "object → time",
              "tense → future", "→ negative"]
    for slot, value in [(0, 0), (1, 4), (2, 4), (3, 2), (4, 1)]:
        mm = base.clone()
        mm[0, slot] = value
        rows.append(mm)
    imgs = paint(speaker, torch.cat(rows), dev, res=480)
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.6), facecolor="white")
    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(to_rgb(img), interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(label, fontsize=10, color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "minimal_pairs.png", dpi=110, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'poster.png'} and {OUT / 'minimal_pairs.png'}")


def say(args):
    dev = H.device()
    speaker, listener = load(dev)
    m = H.parse_sentence(args.words, args.tense, args.negative)
    with torch.no_grad():
        lo = speaker(m.to(dev))
        read = torch.stack([l.argmax(1) for l in listener(H.channel(lo, math.pi, occlude_p=0))], 1).cpu()[0]
    img = paint(speaker, m, dev, res=1024)[0]
    name = "_".join(args.words) + f"_{args.tense}" + ("_neg" if args.negative else "")
    path = OUT / f"say_{name}.png"
    plt.imsave(path, to_rgb(img))
    print(f"wrote {path}\nlistener read (rotated): {H.gloss(read)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--ink-weight", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--soft-until", type=float, default=0.4)
    t.add_argument("--marks-target", type=float, default=3.5)
    t.add_argument("--mark-weight", type=float, default=0.5)
    sub.add_parser("poster")
    s = sub.add_parser("say")
    s.add_argument("words", nargs="+")
    s.add_argument("--tense", default="present", choices=H.TENSES)
    s.add_argument("--negative", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"train": train, "poster": poster, "say": say}[args.cmd](args)


if __name__ == "__main__":
    main()
