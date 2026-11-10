#!/usr/bin/env python3
"""Heptapod Ink v1: an emergent circular script between two neural networks.

A *speaker* receives a whole sentence (subject, verb, object, tense, polarity)
and paints ONE circular ink logogram through a differentiable brush renderer.
A *listener* sees only the (randomly rotated, noisy, partly occluded) image and
must recover the full sentence. Nobody designs the alphabet: it emerges from
the pressure to communicate.

Constraints borrowed from Arrival's Heptapod B (as a design brief, not canon):
  * one logogram = one complete sentence, emitted all at once (no token order)
  * the logogram is a ring of ink with no beginning: the listener sees it at a
    random rotation, so meaning has to live in relative structure
  * ink is costly, so marks stay sparse and deliberate

Usage:
  python heptapod.py train --steps 8000
  python heptapod.py say heptapod offers weapon --tense future
  python heptapod.py say human understands time --negative
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"  # replaced by --out at startup

ENTITIES = ["human", "heptapod", "ship", "weapon", "time", "language"]
VERBS = ["sees", "gives", "fears", "offers", "understands", "becomes"]
TENSES = ["past", "present", "future"]
POLARITY = ["affirmative", "negative"]
SLOTS = {
    "subject": ENTITIES,
    "verb": VERBS,
    "object": ENTITIES,
    "tense": TENSES,
    "polarity": POLARITY,
}
SLOT_NAMES = list(SLOTS)
SIZES = [len(v) for v in SLOTS.values()]

# Canvas and brush
RES = 96
RING_R = 0.58
SECTORS = 32
PARAMS_PER_SECTOR = 5  # ink, width, radial offset, splash amount, splash length


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------- meanings


def all_meanings() -> torch.Tensor:
    grids = torch.meshgrid(*[torch.arange(n) for n in SIZES], indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=1)


def split_meanings(seed: int = 0, held_out_triples: int = 36):
    """Hold out whole (subject, verb, object) triples in every tense/polarity.

    Every individual word is still seen in training; only these specific
    combinations are new. This is the systematic-generalization test.
    """
    rng = random.Random(seed)
    triples = [(s, v, o) for s in range(6) for v in range(6) for o in range(6)]
    rng.shuffle(triples)
    held = set(triples[:held_out_triples])
    meanings = all_meanings()
    is_held = torch.tensor([(int(m[0]), int(m[1]), int(m[2])) in held for m in meanings])
    return meanings[~is_held], meanings[is_held]


def gloss(m) -> str:
    s, v, o, t, p = [int(x) for x in m]
    neg = "not " if p == 1 else ""
    return f"{ENTITIES[s]} {neg}{VERBS[v]} {ENTITIES[o]} ({TENSES[t]})"


def parse_sentence(words: list[str], tense: str, negative: bool) -> torch.Tensor:
    if len(words) != 3:
        raise SystemExit("say expects: SUBJECT VERB OBJECT, e.g. `say heptapod offers weapon`")
    s, v, o = words
    try:
        return torch.tensor([[ENTITIES.index(s), VERBS.index(v), ENTITIES.index(o),
                              TENSES.index(tense), int(negative)]])
    except ValueError as exc:
        raise SystemExit(f"Unknown word. Entities: {ENTITIES}. Verbs: {VERBS}.") from exc


# ---------------------------------------------------------------- renderer


class InkRenderer(nn.Module):
    """Differentiable brush: a ring split into overlapping angular sectors.

    Each sector carries ink amount, stroke width, radial wobble and an outward
    splash (the tendrils of the film's logograms). Output: ink density in [0,1].
    """

    def __init__(self, res: int = RES):
        super().__init__()
        lin = torch.linspace(-1, 1, res)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        rho = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)
        centers = torch.arange(SECTORS) * (2 * math.pi / SECTORS)
        dphi = torch.remainder(phi[None] - centers[:, None, None] + math.pi, 2 * math.pi) - math.pi
        self.register_buffer("rho", rho)
        self.register_buffer("dphi", dphi)  # (K, H, W)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        b = raw.shape[0]
        p = raw.view(b, SECTORS, PARAMS_PER_SECTOR)
        ink = torch.sigmoid(p[..., 0])[..., None, None]
        width = (0.018 + 0.07 * torch.sigmoid(p[..., 1]))[..., None, None]
        offset = (0.12 * torch.tanh(p[..., 2]))[..., None, None]
        splash = torch.sigmoid(p[..., 3])[..., None, None]
        splash_len = (0.05 + 0.3 * torch.sigmoid(p[..., 4]))[..., None, None]

        rho, dphi = self.rho[None, None], self.dphi[None]
        sigma_theta = 0.65 * (2 * math.pi / SECTORS)
        angular = torch.exp(-(dphi**2) / (2 * sigma_theta**2))
        band = ink * angular * torch.exp(-((rho - RING_R - offset) ** 2) / (2 * width**2))

        arc = dphi * rho  # tangential distance
        r0 = RING_R + offset
        radial_window = torch.sigmoid((rho - r0) / 0.012) * torch.sigmoid((r0 + splash_len - rho) / 0.012)
        taper = 1.0 - ((rho - r0) / splash_len).clamp(0, 1) * 0.7
        tendril = ink * splash * radial_window * torch.exp(-(arc**2) / (2 * (0.018 * taper) ** 2))

        density = (band + tendril).sum(dim=1)
        self.last_ink = ink.flatten(1)  # per-sector ink, used for the decisiveness penalty
        return 1 - torch.exp(-6.0 * density)


# ---------------------------------------------------------------- agents


class Speaker(nn.Module):
    def __init__(self, emb: int = 32, hidden: int = 256):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(n, emb) for n in SIZES])
        self.net = nn.Sequential(
            nn.Linear(emb * len(SIZES), hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, SECTORS * PARAMS_PER_SECTOR),
        )
        self.renderer = InkRenderer()

    def brush(self, meanings: torch.Tensor) -> torch.Tensor:
        z = torch.cat([e(meanings[:, i]) for i, e in enumerate(self.embeds)], dim=1)
        return self.net(z)

    def forward(self, meanings: torch.Tensor) -> torch.Tensor:
        return self.renderer(self.brush(meanings))


def meaning_index(meanings: torch.Tensor) -> torch.Tensor:
    idx = torch.zeros(len(meanings), dtype=torch.long, device=meanings.device)
    for i, n in enumerate(SIZES):
        idx = idx * n + meanings[:, i]
    return idx


class HolisticSpeaker(nn.Module):
    """No shared structure at all: every sentence owns a free drawing.

    Each of the 1,296 sentences is a row of free brush parameters. Nothing tells
    the speaker that two sentences share a word; any reuse of marks across
    sentences can only come from the listener's pressure.
    """

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(0.5 * torch.randn(int(np.prod(SIZES)), SECTORS * PARAMS_PER_SECTOR))
        self.renderer = InkRenderer()

    def brush(self, meanings: torch.Tensor) -> torch.Tensor:
        return self.table[meaning_index(meanings)]

    def forward(self, meanings: torch.Tensor) -> torch.Tensor:
        return self.renderer(self.brush(meanings))


def make_speaker(kind: str) -> nn.Module:
    return HolisticSpeaker() if kind == "holistic" else Speaker()


class Listener(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(3)
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(128 * 9, 384), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(384, n) for n in SIZES])

    def forward(self, img: torch.Tensor):
        h = self.trunk(self.pool(self.features(img[:, None])))
        return [head(h) for head in self.heads]


# ---------------------------------------------------------------- channel


BLUR = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16


def channel(img: torch.Tensor, max_angle: float, noise: float = 0.25, occlude_p: float = 0.4):
    """What travels between the agents: rotated, blurred, noisy, sometimes smudged.

    Heavy pixel noise makes subtle grey levels unreliable, so the only robust
    way to talk is with bold, deliberate marks, like brush ink.
    """
    b = img.shape[0]
    ang = (torch.rand(b, device=img.device) * 2 - 1) * max_angle
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                         torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(theta, (b, 1, RES, RES), align_corners=False)
    out = F.grid_sample(img[:, None], grid, align_corners=False, padding_mode="zeros")
    out = F.conv2d(out, BLUR.to(img.device)[None, None], padding=1)[:, 0]
    out = out + noise * torch.randn_like(out)
    if occlude_p > 0:
        mask = torch.ones_like(out)
        for i in torch.nonzero(torch.rand(b) < occlude_p).flatten().tolist():
            cy, cx = random.randrange(8, RES - 24), random.randrange(8, RES - 24)
            mask[i, cy:cy + 16, cx:cx + 16] = 0
        out = out * mask
    return out


# ---------------------------------------------------------------- training


def evaluate(speaker, listener, meanings, dev, rotate=math.pi, repeats=4):
    speaker.eval(), listener.eval()
    per_slot = torch.zeros(len(SIZES))
    exact = 0.0
    n = 0
    with torch.no_grad():
        for _ in range(repeats):
            for chunk in meanings.split(512):
                chunk = chunk.to(dev)
                logits = listener(channel(speaker(chunk), rotate))
                preds = torch.stack([l.argmax(1) for l in logits], 1)
                hit = (preds == chunk).float().cpu()
                per_slot += hit.sum(0)
                exact += hit.all(1).float().sum().item()
                n += len(chunk)
    speaker.train(), listener.train()
    return {"sentence_exact": round(exact / n, 4),
            **{k: round(float(v / n), 4) for k, v in zip(SLOT_NAMES, per_slot)}}


def topographic_similarity(speaker, meanings, dev, pairs=4000, seed=0):
    """Spearman correlation between meaning distance and logogram distance.

    High values mean similar sentences get similar-looking logograms, i.e. the
    script is compositional rather than an arbitrary lookup table.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        imgs = torch.cat([speaker(c.to(dev)).cpu() for c in meanings.split(512)]).flatten(1)
    i = torch.randint(len(meanings), (pairs,), generator=g)
    j = torch.randint(len(meanings), (pairs,), generator=g)
    keep = i != j
    i, j = i[keep], j[keep]
    dm = (meanings[i] != meanings[j]).float().sum(1)
    di = (imgs[i] - imgs[j]).norm(dim=1)

    def rank(x):
        return x.argsort().argsort().float()

    def spearman(a, b):
        ra, rb = rank(a + 1e-6 * torch.randn_like(a)), rank(b)
        ra, rb = ra - ra.mean(), rb - rb.mean()
        return float((ra * rb).sum() / (ra.norm() * rb.norm()))

    real = spearman(dm, di)
    shuffled = spearman(dm[torch.randperm(len(dm), generator=g)], di)
    return round(real, 4), round(shuffled, 4)


def train(args):
    torch.manual_seed(args.seed), random.seed(args.seed), np.random.seed(args.seed)
    dev = device()
    if args.speaker == "holistic":
        # A free drawing per sentence cannot be written for an unseen sentence,
        # so every sentence is trained; there is no held-out split here.
        train_m, held_m = all_meanings(), all_meanings()[:0]
    else:
        train_m, held_m = split_meanings(args.seed)
    speaker, listener = make_speaker(args.speaker).to(dev), Listener().to(dev)
    groups = [{"params": listener.parameters()}]
    if args.speaker == "holistic":
        groups.append({"params": speaker.parameters(), "lr": args.lr * 10, "weight_decay": 0.0})
    else:
        groups.append({"params": speaker.parameters()})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g.get("lr", args.lr) for g in groups],
                                                total_steps=args.steps, pct_start=0.1)
    history = []
    print(f"device={dev} train_meanings={len(train_m)} held_out={len(held_m)}")
    for step in range(1, args.steps + 1):
        # Curriculum: rotation range grows from 0 to the full circle.
        max_angle = math.pi * min(1.0, step / (0.5 * args.steps))
        batch = train_m[torch.randint(len(train_m), (args.batch,))].to(dev)
        img = speaker(batch)
        logits = listener(channel(img, max_angle))
        ce = sum(F.cross_entropy(l, batch[:, i]) for i, l in enumerate(logits))
        ink_cost = img.mean()
        sector_ink = speaker.renderer.last_ink
        indecision = (sector_ink * (1 - sector_ink)).mean()  # push each mark to fully on/off
        loss = ce + args.ink_cost * ink_cost + args.decisive * indecision
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(), sched.step()
        if step % 250 == 0 or step == 1:
            row = {"step": step, "loss": round(ce.item(), 4), "ink": round(ink_cost.item(), 4),
                   "rotation_deg": round(math.degrees(max_angle))}
            history.append(row)
            print(row, flush=True)

    OUT.mkdir(exist_ok=True)
    torch.save({"speaker": speaker.state_dict(), "listener": listener.state_dict(), "seed": args.seed,
                "speaker_kind": args.speaker}, OUT / "model.pt")
    report = {
        "setup": {"speaker": args.speaker, "steps": args.steps, "batch": args.batch, "ink_cost": args.ink_cost, "decisive": args.decisive,
                  "train_meanings": len(train_m), "held_out_meanings": len(held_m),
                  "chance_sentence_exact": round(1 / int(np.prod(SIZES)), 6)},
        "train_meanings_full_rotation": evaluate(speaker, listener, train_m, dev),
    }
    if len(held_m):
        report["held_out_triples_full_rotation"] = evaluate(speaker, listener, held_m, dev)
        report["held_out_no_rotation"] = evaluate(speaker, listener, held_m, dev, rotate=0.0)
    topo, topo_shuf = topographic_similarity(speaker, all_meanings(), dev)
    report["topographic_similarity"] = {"real": topo, "shuffled_baseline": topo_shuf}
    report["word_footprint"] = word_footprint(speaker, dev)
    report["history"] = history
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    figures(speaker, dev, held_m)


@torch.no_grad()
def footprint_maps(speaker, dev, samples: int = 300, seed: int = 0):
    """For each slot: average |ink change| when only that word changes."""
    speaker.eval()
    g = torch.Generator().manual_seed(seed)
    base = torch.stack([torch.randint(n, (samples,), generator=g) for n in SIZES], 1)
    maps = []
    for slot, n in enumerate(SIZES):
        alt = base.clone()
        alt[:, slot] = (base[:, slot] + torch.randint(1, n, (samples,), generator=g)) % n
        d = (speaker(base.to(dev)) - speaker(alt.to(dev))).abs().mean(0).cpu()
        maps.append(d)
    return torch.stack(maps)


def word_footprint(speaker, dev):
    """How much of the ring does one word repaint?

    `ring_share` is the share of the ring's angle that carries 80% of the ink
    change: low = the word owns a zone, high (up to 1.0) = the whole ring is
    repainted, like the film's logograms. `ink_changed` is the mean absolute
    change in ink inside the ring band.
    """
    maps = footprint_maps(speaker, dev)
    lin = torch.linspace(-1, 1, RES)
    y, x = torch.meshgrid(lin, lin, indexing="ij")
    rho, phi = torch.sqrt(x**2 + y**2), torch.atan2(y, x)
    band = (rho > RING_R - 0.15) & (rho < RING_R + 0.4)
    bins = ((phi + math.pi) / (2 * math.pi) * 72).long().clamp(0, 71)
    out = {}
    for name, m in zip(SLOT_NAMES, maps):
        prof = torch.zeros(72).index_add_(0, bins[band], m[band])
        sorted_prof = prof.sort(descending=True).values
        k = int((sorted_prof.cumsum(0) < 0.8 * prof.sum()).sum()) + 1
        out[name] = {"ring_share": round(k / 72, 3), "ink_changed": round(float(m[band].mean()), 4)}
    return out


def footprint_figure(speaker, dev, path: Path, title: str):
    maps = footprint_maps(speaker, dev)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), facecolor=PAPER)
    vmax = float(maps.max())
    for ax, name, m in zip(axes, SLOT_NAMES, maps):
        ax.imshow(m, cmap="magma", vmin=0, vmax=vmax)
        ax.set_axis_off()
        ax.set_title(f"change the {name}", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=PAPER)
    plt.close(fig)


# ---------------------------------------------------------------- figures

PAPER = "#d9d6cf"
INK = "#121212"


def show(ax, img, title=None, size=9):
    rgb_paper = np.array(matplotlib.colors.to_rgb(PAPER))
    rgb_ink = np.array(matplotlib.colors.to_rgb(INK))
    a = np.asarray(img)[..., None]
    ax.imshow(rgb_paper * (1 - a) + rgb_ink * a, interpolation="bicubic")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=size, color="#333")


@torch.no_grad()
def render(speaker, meanings, dev):
    speaker.eval()
    return speaker(meanings.to(dev)).cpu()


def figures(speaker, dev, held_m):
    # 1) Minimal pairs: change one slot at a time.
    base = torch.tensor([[1, 3, 3, 0, 0]])  # heptapod offers weapon (past)
    rows = [base]
    variations = [(0, 0, "subject → human"), (1, 4, "verb → understands"),
                  (2, 4, "object → time"), (3, 2, "tense → future"), (4, 1, "→ negative")]
    for slot, value, _ in variations:
        m = base.clone()
        m[0, slot] = value
        rows.append(m)
    imgs = render(speaker, torch.cat(rows), dev)
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6), facecolor=PAPER)
    show(axes[0, 0], imgs[0], gloss(base[0]))
    axes[1, 0].set_axis_off()
    for k, (_, _, label) in enumerate(variations, start=1):
        show(axes[0, k], imgs[k], label)
        diff = (imgs[k] - imgs[0]).numpy()
        axes[1, k].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1, k].set_axis_off()
        axes[1, k].set_title("red = ink added, blue = removed", fontsize=7, color="#555")
    fig.suptitle("Minimal pairs: what changes in the logogram when one word changes", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "minimal_pairs.png", dpi=130, facecolor=PAPER)
    plt.close(fig)

    # 2) Lexicon grid: subjects × verbs, fixed object/tense/polarity.
    grid = torch.tensor([[s, v, 3, 1, 0] for s in range(6) for v in range(6)])
    imgs = render(speaker, grid, dev)
    fig, axes = plt.subplots(6, 6, figsize=(12, 12.6), facecolor=PAPER)
    for idx, ax in enumerate(axes.flat):
        s, v = idx // 6, idx % 6
        show(ax, imgs[idx], f"{ENTITIES[s]} {VERBS[v]}", size=8)
    fig.suptitle("Subject × verb  (object = weapon, present, affirmative)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "lexicon_grid.png", dpi=110, facecolor=PAPER)
    plt.close(fig)

    footprint_figure(speaker, dev, OUT / "word_footprint.png",
                     "Where each word lives: bright = ink that changes when only that word changes")

    # 3) Held-out sentences the speaker never wrote during training.
    if not len(held_m):
        return
    sample = held_m[torch.randperm(len(held_m), generator=torch.Generator().manual_seed(1))[:8]]
    imgs = render(speaker, sample, dev)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), facecolor=PAPER)
    for ax, img, m in zip(axes.flat, imgs, sample):
        show(ax, img, gloss(m), size=9)
    fig.suptitle("Sentences never seen in training", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "held_out.png", dpi=120, facecolor=PAPER)
    plt.close(fig)


def load(dev):
    ckpt = torch.load(OUT / "model.pt", map_location=dev)
    speaker, listener = make_speaker(ckpt.get("speaker_kind", "compositional")).to(dev), Listener().to(dev)
    speaker.load_state_dict(ckpt["speaker"])
    listener.load_state_dict(ckpt["listener"])
    return speaker.eval(), listener.eval()


def say(args):
    dev = device()
    speaker, listener = load(dev)
    m = parse_sentence(args.words, args.tense, args.negative)
    img = render(speaker, m, dev)
    with torch.no_grad():
        logits = listener(channel(img.to(dev), math.pi, occlude_p=0.0))
    read = torch.stack([l.argmax(1) for l in logits], 1).cpu()[0]
    fig, ax = plt.subplots(figsize=(6, 6.4), facecolor=PAPER)
    show(ax, img[0], gloss(m[0]), size=12)
    ax.text(0.5, -0.04, f"listener read (rotated): {gloss(read)}", transform=ax.transAxes,
            ha="center", fontsize=9, color="#555")
    name = "_".join(args.words) + f"_{args.tense}" + ("_neg" if args.negative else "")
    path = OUT / f"say_{name}.png"
    fig.savefig(path, dpi=140, facecolor=PAPER, bbox_inches="tight")
    print(f"wrote {path}\nlistener read: {gloss(read)}")


def analyze(args):
    dev = device()
    speaker, _ = load(dev)
    fp = word_footprint(speaker, dev)
    metrics_path = OUT / "metrics.json"
    report = json.loads(metrics_path.read_text())
    report["word_footprint"] = fp
    metrics_path.write_text(json.dumps(report, indent=2))
    footprint_figure(speaker, dev, OUT / "word_footprint.png",
                     "Where each word lives: bright = ink that changes when only that word changes")
    print(json.dumps(fp, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--ink-cost", type=float, default=2.0)
    t.add_argument("--decisive", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--speaker", default="compositional", choices=["compositional", "holistic"])
    s = sub.add_parser("say")
    s.add_argument("words", nargs="+")
    s.add_argument("--tense", default="present", choices=TENSES)
    s.add_argument("--negative", action="store_true")
    sub.add_parser("analyze")
    ap.add_argument("--out", default="out", help="output folder inside v1/")
    args = ap.parse_args()
    global OUT
    OUT = HERE / args.out
    OUT.mkdir(exist_ok=True)
    {"train": train, "say": say, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
