"""Measure the visual statistics of logograms (reference or generated).

Each image is binarized, centered on its ring, and described by:
  ink: share of the canvas covered by ink
  ring_radius: radius of the ring relative to the canvas half-width
  ring_width: median thickness of the bare ring (where there is no cluster)
  gap_share: share of the ring's angle with no ink at all (open ring)
  cluster_share: share of the ring's angle carrying heavy ink (masses)
  clusters: number of separate heavy-ink runs around the ring
"""
import json, sys, glob
import numpy as np
from PIL import Image


def describe(a: np.ndarray) -> dict:
    ink = a > 0.5
    n = ink.shape[0]
    yy, xx = np.nonzero(ink)
    cy, cx = yy.mean(), xx.mean()
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(y - cy, x - cx) / (n / 2)
    phi = np.arctan2(y - cy, x - cx)
    hist, edges = np.histogram(r[ink], bins=200, range=(0, 1.5))
    ring_r = edges[hist.argmax()]
    bins = ((phi + np.pi) / (2 * np.pi) * 180).astype(int).clip(0, 179)
    band = np.abs(r - ring_r) < 0.06
    near = np.abs(r - ring_r) < 0.35
    mass = np.bincount(bins[near & ink], minlength=180).astype(float)
    on_ring = np.bincount(bins[band & ink], minlength=180)
    bare = mass[mass > 0]
    base = np.median(bare) if len(bare) else 0
    heavy = mass > 3 * max(base, 1)
    runs = np.sum(heavy & ~np.roll(heavy, 1)) if heavy.any() and not heavy.all() else int(heavy.all())
    radial_px = mass / ((n / 2) * (2 * np.pi / 180) * ring_r)  # ink pixels per arc pixel
    return {
        "ink": float(ink.mean()),
        "ring_radius": float(ring_r),
        "ring_width": float(np.median(radial_px[(mass > 0) & ~heavy]) / (n / 2)) if ((mass > 0) & ~heavy).any() else 0.0,
        "gap_share": float((on_ring == 0).mean()),
        "cluster_share": float(heavy.mean()),
        "clusters": int(runs),
    }


def summarize(rows):
    keys = rows[0].keys()
    return {k: {"median": round(float(np.median([r[k] for r in rows])), 4),
                "p10": round(float(np.percentile([r[k] for r in rows], 10)), 4),
                "p90": round(float(np.percentile([r[k] for r in rows], 90)), 4)} for k in keys}


def load_ref(path, size=512):
    g = np.asarray(Image.open(path).convert("L").resize((size, size), Image.LANCZOS)) / 255.0
    return 1 - g  # ink = 1


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "logograms", "*.jpg")))
    rows = [describe(load_ref(f)) for f in files]
    out = summarize(rows)
    print(json.dumps(out, indent=2))
    json.dump({"summary": out, "per_image": dict(zip([f.split('/')[-1] for f in files], rows))},
              open(os.path.join(here, "reference_stats.json"), "w"), indent=2)
