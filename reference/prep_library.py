"""Turn the reference logograms into a library of real ink fragments.

1. Fit the ring of every logogram (RANSAC circle fit on ink pixels).
2. Unwrap it into a polar strip: angle across, radius (relative to the ring) down.
   In that space every logogram has its ring on the same row, so pieces from
   different logograms line up when placed next to each other.
3. Cut sliding angular windows ("pie slices") from every strip.

Output: fragments.npz with high-res and low-res polar fragments (ink = 1).
"""
import glob, json, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SIZE = 2048
R_IN, R_OUT = 0.45, 1.65        # radial range relative to the fitted ring radius
ANG_HI, RAD_HI = 3600, 420       # full-circle polar strip, high res
SLOTS = 12
WIDTH_DEG = 40                   # fragment width (30 deg slot + feathered overlap)
STRIDE_DEG = 15
LO_ANG_PER_DEG = 0.5             # low-res strip: 180 columns per circle
LO_RAD = 34


def fit_ring(ink, iters=3000, seed=0):
    rng = np.random.default_rng(seed)
    n = ink.shape[0]
    ys, xs = np.nonzero(ink)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    sub = pts[rng.choice(len(pts), min(len(pts), 20000), replace=False)]
    best, best_score = None, -1
    for _ in range(iters):
        p = sub[rng.choice(len(sub), 3, replace=False)]
        (x1, y1), (x2, y2), (x3, y3) = p
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d) < 1e-6:
            continue
        ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
        uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
        r = math.hypot(x1 - ux, y1 - uy)
        if not (0.25 * n / 2 < r < 0.9 * n / 2):
            continue
        dist = np.abs(np.hypot(sub[:, 0] - ux, sub[:, 1] - uy) - r)
        inl = dist < 0.006 * n
        # Reward circles whose inliers cover many angles (the ring), not one big blob.
        ang = np.arctan2(sub[inl, 1] - uy, sub[inl, 0] - ux)
        coverage = len(np.unique(((ang + np.pi) / (2 * np.pi) * 90).astype(int)))
        score = coverage * 1000 + inl.sum()
        if score > best_score:
            best, best_score = (ux, uy, r), score
    return best


def unwrap(ink_t, cx, cy, r):
    n = ink_t.shape[-1]
    theta = torch.linspace(-math.pi, math.pi, ANG_HI + 1)[:-1]
    rad = torch.linspace(R_IN, R_OUT, RAD_HI) * r
    tt, rr = torch.meshgrid(theta, rad, indexing="xy")  # (RAD, ANG)
    px = cx + rr * torch.cos(tt)
    py = cy + rr * torch.sin(tt)
    grid = torch.stack([px / (n - 1) * 2 - 1, py / (n - 1) * 2 - 1], -1)[None].float()
    return F.grid_sample(ink_t[None, None], grid, align_corners=True, padding_mode="zeros")[0, 0]


def main():
    files = sorted(glob.glob("logograms/*.jpg"))
    strips, fits = [], {}
    for f in files:
        g = np.asarray(Image.open(f).convert("L").resize((SIZE, SIZE), Image.LANCZOS)) / 255.0
        ink = 1 - g
        if "WhereIsAbbott" in f:  # this file carries an English caption, bottom left
            ink[int(0.92 * SIZE):, : int(0.45 * SIZE)] = 0
        small = ink[::4, ::4] > 0.5
        cx, cy, r = [v * 4 for v in fit_ring(small)]
        fits[f.split("/")[-1]] = {"cx": cx / SIZE, "cy": cy / SIZE, "r": r / SIZE}
        strips.append(unwrap(torch.tensor(ink, dtype=torch.float32), cx, cy, r))
        print(f, fits[f.split("/")[-1]], flush=True)
    strips = torch.stack(strips)  # (38, RAD_HI, ANG_HI)

    w_hi = int(ANG_HI * WIDTH_DEG / 360)
    stride_hi = int(ANG_HI * STRIDE_DEG / 360)
    frags_hi, origin = [], []
    for i, s in enumerate(strips):
        wrapped = torch.cat([s, s[:, :w_hi]], 1)
        for start in range(0, ANG_HI, stride_hi):
            frags_hi.append(wrapped[:, start:start + w_hi])
            origin.append((files[i].split("/")[-1], start * 360 / ANG_HI))
    frags_hi = torch.stack(frags_hi)
    w_lo = int(WIDTH_DEG * LO_ANG_PER_DEG)
    frags_lo = F.interpolate(frags_hi[:, None], size=(LO_RAD, w_lo), mode="area")[:, 0]
    np.savez_compressed("fragments.npz",
                        hi=(frags_hi.numpy() * 255).astype(np.uint8),
                        lo=frags_lo.numpy().astype(np.float32),
                        strips=(F.interpolate(strips[:, None], size=(210, 1800), mode="area")[:, 0].numpy() * 255).astype(np.uint8),
                        r_in=R_IN, r_out=R_OUT, width_deg=WIDTH_DEG, slots=SLOTS)
    json.dump({"fits": fits, "origin": origin}, open("fragments_index.json", "w"), indent=1)
    print("fragments", frags_hi.shape, frags_lo.shape)


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # inputs/outputs live next to this script
    main()
