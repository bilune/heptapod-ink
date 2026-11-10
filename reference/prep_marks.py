"""Split the reference logograms into whole marks and clean ring segments.

Works on the polar strips made by prep_library.py (ring on a fixed row).
  marks: every contiguous run of angles with ink off the ring line (a cluster,
         with its tendrils and droplets), cut where the ring is quiet, so a
         mark is never sliced through.
  rings: 40-degree windows with nothing but the ring line (thin, thick, wobbly),
         plus implicit "gap" (no ring) handled by the model.
"""
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # inputs/outputs live next to this script
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F

lib = np.load("fragments.npz")
R_IN, R_OUT = float(lib["r_in"]), float(lib["r_out"])
import prep_library as P

# Rebuild full-res strips (the npz keeps only a reduced preview of them).
import glob
from PIL import Image

files = sorted(glob.glob("logograms/*.jpg"))
fits = json.load(open("fragments_index.json"))["fits"]
strips = []
for f in files:
    g = np.asarray(Image.open(f).convert("L").resize((P.SIZE, P.SIZE), Image.LANCZOS)) / 255.0
    ink = 1 - g
    if "WhereIsAbbott" in f:
        ink[int(0.92 * P.SIZE):, : int(0.45 * P.SIZE)] = 0
    fit = fits[f.split("/")[-1]]
    strips.append(P.unwrap(torch.tensor(ink, dtype=torch.float32), fit["cx"] * P.SIZE, fit["cy"] * P.SIZE, fit["r"] * P.SIZE))
strips = torch.stack(strips)  # (38, 420, 3600), 10 columns per degree
N, RAD, COLS = strips.shape
CPD = COLS // 360
r = torch.linspace(R_IN, R_OUT, RAD)
off = (r - 1).abs() > 0.07

MAX_MARK_DEG = 120
marks, mark_origin, rings, ring_origin = [], [], [], []
for i, s in enumerate(strips):
    act = s[off].mean(0)  # off-ring ink per column
    active = (act > 0.004).float()
    # close small holes (< 2 deg) so one splash stays one mark
    k = 2 * CPD + 1
    active = (F.max_pool1d(active[None, None], k, 1, k // 2)[0, 0] > 0).float()
    active = (-F.max_pool1d(-active[None, None], k, 1, k // 2)[0, 0] > 0)
    if active.all():
        continue
    # rotate the strip so it starts on a quiet column (runs never wrap)
    start = int(torch.nonzero(~active)[0])
    s2, a2 = torch.roll(s, -start, 1), torch.roll(active, -start, 0)
    c = 0
    while c < COLS:
        if a2[c]:
            e = c
            while e < COLS and a2[e]:
                e += 1
            lo, hi = max(0, c - CPD), min(COLS, e + CPD)
            width_deg = (hi - lo) / CPD
            if 3 <= width_deg <= MAX_MARK_DEG:
                piece = torch.zeros(RAD, MAX_MARK_DEG * CPD)
                off0 = (MAX_MARK_DEG * CPD - (hi - lo)) // 2
                piece[:, off0:off0 + hi - lo] = s2[:, lo:hi]
                marks.append(piece)
                mark_origin.append((files[i].split("/")[-1], round(((lo + start) % COLS) / CPD, 1), round(width_deg, 1)))
            c = e
        else:
            c += 1
    # clean ring windows: 40 deg with no off-ring activity
    w = 40 * CPD
    for st in range(0, COLS, 10 * CPD):
        idx = (torch.arange(w) + st) % COLS
        if not active[idx].any() and s[~off][:, idx].mean() > 0.02:
            rings.append(s[:, idx])
            ring_origin.append((files[i].split("/")[-1], st / CPD))

marks = torch.stack(marks)
rings = torch.stack(rings)
print("marks", marks.shape, "rings", rings.shape)
widths = [m[2] for m in mark_origin]
print("mark widths deg: median", np.median(widths), "p90", np.percentile(widths, 90))
LO_CPD = 0.5
lo_rad = 34
np.savez_compressed(
    "marks.npz",
    marks_hi=(marks.numpy() * 255).astype(np.uint8),
    rings_hi=(rings.numpy() * 255).astype(np.uint8),
    marks_lo=F.interpolate(marks[:, None], size=(lo_rad, int(MAX_MARK_DEG * LO_CPD)), mode="area")[:, 0].numpy(),
    rings_lo=F.interpolate(rings[:, None], size=(lo_rad, int(40 * LO_CPD)), mode="area")[:, 0].numpy(),
    r_in=R_IN, r_out=R_OUT, max_mark_deg=MAX_MARK_DEG, ring_deg=40, hi_cpd=CPD, lo_cpd=LO_CPD,
)
json.dump({"marks": mark_origin, "rings": ring_origin}, open("marks_index.json", "w"), indent=1)
