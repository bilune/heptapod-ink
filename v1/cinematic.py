#!/usr/bin/env python3
"""Cinematic renderer: paint the learned script in the visual style of the film.

The speaker's brush parameters (the actual "words" the networks agreed on) are
kept exactly as learned. Only the calligraphy changes: dense smoky ink, an
irregular thick ring, blots, curling tendrils, droplets, on hazy white glass.

This imitates a STYLE. It does not copy or use any of the film's logograms.

Meaning-free details (wobble, curl direction, droplets, texture) are seeded by
the sentence, so the same sentence always looks the same.

Usage:
  python cinematic.py check                       # can the listener still read it?
  python cinematic.py finetune                    # adapt the listener to the new calligraphy
  python cinematic.py poster                      # gallery of sentences
  python cinematic.py say heptapod offers weapon --tense future
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import heptapod as H

# ---------------------------------------------------------------- noise


def fbm(res: int, gen: torch.Generator, octaves=(4, 8, 16, 32), falloff=0.55) -> torch.Tensor:
    """Fractal value noise in [-1, 1], shape (res, res)."""
    total, amp, norm = torch.zeros(res, res), 1.0, 0.0
    for o in octaves:
        g = torch.rand(1, 1, o + 3, o + 3, generator=gen) * 2 - 1
        up = F.interpolate(g, size=(res + res // o * 3, res + res // o * 3), mode="bicubic", align_corners=False)
        total += amp * up[0, 0, :res, :res]
        norm += amp
        amp *= falloff
    return total / norm


def blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.3:
        return img
    k = int(sigma * 3) * 2 + 1
    x = torch.arange(k) - k // 2
    g = torch.exp(-(x.float() ** 2) / (2 * sigma**2))
    g = (g / g.sum())
    t = img[None, None]
    t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    t = F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))
    return t[0, 0]


# ---------------------------------------------------------------- painter


def decode_brush(raw: torch.Tensor) -> dict:
    p = raw.view(H.SECTORS, H.PARAMS_PER_SECTOR)
    return {
        "ink": torch.sigmoid(p[:, 0]),
        "width": 0.018 + 0.07 * torch.sigmoid(p[:, 1]),
        "offset": 0.12 * torch.tanh(p[:, 2]),
        "splash": torch.sigmoid(p[:, 3]),
        "splash_len": 0.05 + 0.3 * torch.sigmoid(p[:, 4]),
    }


def paint(raw: torch.Tensor, seed: int, res: int = 768) -> torch.Tensor:
    """Ink density in [0, 1] at `res`, film-like calligraphy of the same brush."""
    b = decode_brush(raw.detach().float().cpu())
    gen = torch.Generator().manual_seed(seed)
    scale = res / 96.0

    lin = torch.linspace(-1, 1, res)
    y, x = torch.meshgrid(lin, lin, indexing="ij")
    # Domain warp: organic, wet edges instead of geometric ones.
    wx, wy = fbm(res, gen), fbm(res, gen)
    xw, yw = x + 0.022 * wx, y + 0.022 * wy
    rho = torch.sqrt(xw**2 + yw**2)
    phi = torch.atan2(yw, xw)

    centers = torch.arange(H.SECTORS) * (2 * math.pi / H.SECTORS)
    sigma_theta = 0.65 * (2 * math.pi / H.SECTORS)
    density = torch.zeros(res, res)

    # Meaning-free ring wobble: the circle is hand-drawn, not a compass.
    wobble_phase = torch.rand(3, generator=gen) * 2 * math.pi
    wobble = sum(0.012 / (i + 1) * torch.sin((i + 2) * phi + wobble_phase[i]) for i in range(3))

    # A thin continuous base stroke keeps the ring closed, as in the film.
    base_r = H.RING_R + wobble
    # Thickness swells and thins along the ring, like a loaded brush.
    thick_phase = torch.rand(2, generator=gen) * 2 * math.pi
    base_w = 0.020 * (1 + 0.35 * torch.sin(3 * phi + thick_phase[0]) * torch.sin(5 * phi + thick_phase[1]))
    density += 1.0 * torch.exp(-((rho - base_r) ** 2) / (2 * base_w**2))

    for k in range(H.SECTORS):
        ink = float(b["ink"][k])
        if ink < 0.04:
            continue
        dphi = torch.remainder(phi - centers[k] + math.pi, 2 * math.pi) - math.pi
        ang = torch.exp(-(dphi**2) / (2 * sigma_theta**2))
        w = float(b["width"][k]) * 1.35
        r0 = H.RING_R + float(b["offset"][k])
        # Heavier ink pools toward the outside of a thick stroke, like a brush pressed down.
        band = ink * ang * torch.exp(-((rho - r0 - wobble) ** 2) / (2 * w**2))
        density += 2.4 * band

    # Tendrils: the learned splashes, drawn as tapered curling strokes.
    tips = []
    for k in range(H.SECTORS):
        amount = float(b["ink"][k] * b["splash"][k])
        if amount < 0.12:
            continue
        L = float(b["splash_len"][k]) * 1.05
        r0 = H.RING_R + float(b["offset"][k]) - 0.01
        curl = (float(torch.rand(1, generator=gen)) - 0.5) * 0.9
        n = 40
        t = torch.linspace(0, 1, n)
        pr = r0 + L * t
        pa = centers[k] + curl * t**1.6
        px, py = pr * torch.cos(pa), pr * torch.sin(pa)
        w0 = 0.030 * (0.6 + amount)
        widths = w0 * (1 - 0.88 * t) + 0.0015
        # distance from every pixel to the polyline, segment by segment
        ax_, ay_ = px[:-1], py[:-1]
        bx_, by_ = px[1:], py[1:]
        vx, vy = bx_ - ax_, by_ - ay_
        seg_len2 = vx**2 + vy**2 + 1e-9
        gx, gy = xw[None] - ax_[:, None, None], yw[None] - ay_[:, None, None]
        u = ((gx * vx[:, None, None] + gy * vy[:, None, None]) / seg_len2[:, None, None]).clamp(0, 1)
        dx = gx - u * vx[:, None, None]
        dy = gy - u * vy[:, None, None]
        d2 = dx**2 + dy**2
        wseg = widths[:-1][:, None, None] * (1 - u) + widths[1:][:, None, None] * u
        stroke = torch.exp(-d2 / (2 * wseg**2)).amax(0)
        density += 2.0 * min(1.0, amount * 1.6) * stroke
        tips.append((float(px[-1]), float(py[-1]), float(pa[-1]), amount))

    # Droplets flung past some tips.
    for tx, ty, ta, amount in tips:
        for _ in range(int(torch.randint(0, 3, (1,), generator=gen))):
            dist = 0.03 + 0.08 * float(torch.rand(1, generator=gen))
            jitter = (float(torch.rand(1, generator=gen)) - 0.5) * 0.25
            cx, cy = tx + dist * math.cos(ta + jitter), ty + dist * math.sin(ta + jitter)
            rad = 0.004 + 0.008 * float(torch.rand(1, generator=gen)) * amount
            density += 1.2 * torch.exp(-((xw - cx) ** 2 + (yw - cy) ** 2) / (2 * rad**2))

    # Ink saturates: wherever there is real paint it is solid black, with a
    # short wet falloff at the edges instead of a grey gradient.
    soft = 1 - torch.exp(-2.5 * density)
    core = ((soft - 0.18) / 0.32).clamp(0, 1)
    core = core * core * (3 - 2 * core)
    texture = fbm(res, gen, octaves=(16, 32, 64, 128), falloff=0.7)
    core = (core * (0.94 + 0.06 * texture)).clamp(0, 1)
    # Smoke: soft diffusion around the ink, broken up by turbulence.
    turb = (0.5 + 0.5 * fbm(res, gen, octaves=(6, 12, 24))).clamp(0, 1)
    smoke = blur(core, 5 * scale) * turb
    haze = blur(core, 16 * scale)
    ink = (core + 0.28 * smoke + 0.10 * haze).clamp(0, 1)
    return ink


def composite(ink: torch.Tensor, seed: int) -> np.ndarray:
    """Black ink on bright, hazy glass."""
    res = ink.shape[0]
    gen = torch.Generator().manual_seed(seed + 99)
    lin = torch.linspace(-1, 1, res)
    y, x = torch.meshgrid(lin, lin, indexing="ij")
    r = torch.sqrt(x**2 + y**2)
    glow = (1 - 0.22 * r.clamp(0, 1.4) ** 2)
    fog = 0.04 * fbm(res, gen, octaves=(2, 4, 8))
    paper = (0.93 * glow + fog).clamp(0, 1)
    bg = torch.stack([paper * 0.985, paper * 0.99, paper * 1.0], -1)
    ink_rgb = torch.tensor([0.035, 0.035, 0.04])
    a = ink[..., None]
    img = bg * (1 - a) + ink_rgb * a
    grain = 0.012 * torch.randn(res, res, 1, generator=gen)
    return (img + grain).clamp(0, 1).numpy()


# ---------------------------------------------------------------- reading test


def cinematic_batch(speaker, meanings, dev, res=H.RES):
    with torch.no_grad():
        raws = speaker.brush(meanings.to(dev)).cpu()
    return torch.stack([paint(r, int(H.meaning_index(m[None])[0]), res=res) for r, m in zip(raws, meanings)])


def read_accuracy(listener, imgs, meanings, dev, repeats=3):
    listener.eval()
    hits = torch.zeros(len(H.SIZES))
    exact, n = 0.0, 0
    with torch.no_grad():
        for _ in range(repeats):
            for ic, mc in zip(imgs.split(256), meanings.split(256)):
                logits = listener(H.channel(ic.to(dev), math.pi))
                pred = torch.stack([l.argmax(1) for l in logits], 1).cpu()
                hit = pred == mc
                hits += hit.float().sum(0)
                exact += hit.all(1).float().sum().item()
                n += len(mc)
    return {"sentence_exact": round(exact / n, 4), **{k: round(float(v / n), 4) for k, v in zip(H.SLOT_NAMES, hits)}}


def cache_path():
    return H.OUT / "cinematic_96.pt"


def get_cache(speaker, dev):
    path = cache_path()
    if path.exists():
        return torch.load(path)
    m = H.all_meanings()
    print(f"painting {len(m)} sentences at 96px (cached afterwards)...", flush=True)
    imgs = cinematic_batch(speaker, m, dev)
    torch.save({"meanings": m, "imgs": imgs}, path)
    return {"meanings": m, "imgs": imgs}


def check(args):
    dev = H.device()
    speaker, listener = H.load(dev)
    cache = get_cache(speaker, dev)
    m, imgs = cache["meanings"], cache["imgs"]
    simple = torch.cat([speaker(c.to(dev)).cpu() for c in m.split(256)])
    report = {
        "listener_on_training_calligraphy": read_accuracy(listener, simple, m, dev),
        "listener_on_cinematic_calligraphy": read_accuracy(listener, imgs, m, dev),
    }
    print(json.dumps(report, indent=2))
    return report


def finetune(args):
    """Teach the listener the new calligraphy. The speaker is frozen: the
    script itself does not change, only how it is drawn."""
    dev = H.device()
    speaker, listener = H.load(dev)
    cache = get_cache(speaker, dev)
    m, imgs = cache["meanings"], cache["imgs"]
    train_m, held_m = H.split_meanings(0)
    held_set = set(H.meaning_index(held_m).tolist())
    idx = H.meaning_index(m)
    train_mask = torch.tensor([int(i) not in held_set for i in idx])
    before = read_accuracy(listener, imgs[~train_mask], m[~train_mask], dev)
    listener.train()
    opt = torch.optim.AdamW(listener.parameters(), lr=5e-4)
    tm, ti = m[train_mask], imgs[train_mask]
    simple = torch.cat([speaker(c.to(dev)).cpu() for c in tm.split(256)])
    for step in range(1, args.steps + 1):
        sel = torch.randint(len(tm), (128,))
        # Mix both calligraphies so the listener reads either.
        batch_img = torch.where(torch.rand(128, 1, 1) < 0.75, ti[sel], simple[sel]).to(dev)
        logits = listener(H.channel(batch_img, math.pi))
        y = tm[sel].to(dev)
        loss = sum(F.cross_entropy(l, y[:, i]) for i, l in enumerate(logits))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print({"step": step, "loss": round(loss.item(), 4)}, flush=True)
    after = read_accuracy(listener, imgs[~train_mask], m[~train_mask], dev)
    ckpt = torch.load(H.OUT / "model.pt", map_location=dev)
    ckpt["listener"] = listener.state_dict()
    torch.save(ckpt, H.OUT / "model.pt")
    report = {"held_out_cinematic_before_finetune": before, "held_out_cinematic_after_finetune": after}
    metrics = json.loads((H.OUT / "metrics.json").read_text())
    metrics["cinematic"] = report
    (H.OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------- pictures


def save_single(img: np.ndarray, path: Path, caption: str | None = None):
    fig = plt.figure(figsize=(8, 8), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()
    if caption:
        ax.text(0.5, 0.04, caption, transform=ax.transAxes, ha="center", fontsize=11, color="#555")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def poster(args):
    dev = H.device()
    speaker, _ = H.load(dev)
    sentences = [
        (1, 3, 3, 2, 0), (0, 4, 4, 1, 0), (1, 5, 5, 0, 0),
        (0, 2, 1, 1, 1), (2, 1, 0, 2, 0), (5, 4, 4, 2, 0),
    ]
    m = torch.tensor(sentences)
    with torch.no_grad():
        raws = speaker.brush(m.to(dev)).cpu()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.6), facecolor="#e9e8e4")
    for ax, raw, mm in zip(axes.flat, raws, m):
        seed = int(H.meaning_index(mm[None])[0])
        ax.imshow(composite(paint(raw, seed, res=640), seed), interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(H.gloss(mm), fontsize=11, color="#444")
    fig.tight_layout()
    path = H.OUT / "cinematic_poster.png"
    fig.savefig(path, dpi=110, facecolor="#e9e8e4")
    print(f"wrote {path}")


def say(args):
    dev = H.device()
    speaker, _ = H.load(dev)
    m = H.parse_sentence(args.words, args.tense, args.negative)
    with torch.no_grad():
        raw = speaker.brush(m.to(dev)).cpu()[0]
    seed = int(H.meaning_index(m)[0])
    img = composite(paint(raw, seed, res=1024), seed)
    name = "_".join(args.words) + f"_{args.tense}" + ("_neg" if args.negative else "")
    path = H.OUT / f"cinematic_{name}.png"
    save_single(img, path, caption=H.gloss(m[0]) if args.caption else None)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    f = sub.add_parser("finetune")
    f.add_argument("--steps", type=int, default=1500)
    sub.add_parser("poster")
    s = sub.add_parser("say")
    s.add_argument("words", nargs="+")
    s.add_argument("--tense", default="present", choices=H.TENSES)
    s.add_argument("--negative", action="store_true")
    s.add_argument("--caption", action="store_true")
    args = ap.parse_args()
    H.OUT = H.HERE / args.out
    {"check": check, "finetune": finetune, "poster": poster, "say": say}[args.cmd](args)


if __name__ == "__main__":
    main()
