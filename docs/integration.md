# Integration: drawing rings with your own renderer

This page is for anyone who wants to draw Heptapod Ink logograms their own way (an animation,
a plotter, a stylised renderer) while keeping them readable. The model decides *which mark goes in
which slot*; your renderer decides how it looks.

## What a logogram is

A logogram is 36 integers: one ink mark id per slot.

- **Slots.** 36 slots around a fixed ring, 10 degrees each. Slot `k` is centred at angle
  `(k + 0.5) * 10` degrees.
- **Angles.** Image coordinates: x to the right, y down, `phi = atan2(y, x)`, starting at
  `phi = -180` degrees. So slot 0 begins at 9 o'clock and the slots advance clockwise
  (9 -> 12 -> 3 -> 6 o'clock).
- **Marks.** Each slot holds a mark id from 0 to 83; 0 is blank. Only 81 ids are ever used (the
  symbol-to-mark table is `model.fsq_to_mark`, also `sym_to_mark` in `web/models/config.json`).
  Marks are about 30 degrees wide, so each one overlaps its neighbours.
- **Ring line.** The same in every slot and every logogram. It carries no information.
- **Ink library.** `reference/marks.npz`: `marks_hi` holds the 83 cut marks, `rings_hi` the ring
  segments. Cut from the Arrival logograms published by Wolfram Research (CC BY-NC 4.0).
- **Reference renderer.** `Script` in [v4/heptapod_t5.py](../v4/heptapod_t5.py), built on `RingMarkRenderer` from
  [v2/heptapod_ink.py](../v2/heptapod_ink.py). Training resolution 288 px, mark scale 0.25,
  ring at radius 0.5 of the [-1, 1] canvas.

## Getting the marks for a sentence

Run from the repo root, with the weights in `v4/runs/s2/` (see the [README](../README.md#weights)).

```python
import sys; sys.path.insert(0, "v4")
from pathlib import Path
import heptapod_t5 as H
import search as SR

m, _ = H.load(Path("v4/runs/s2/model.pt"), H.device()); m.eval()

# quick: the writer's first ring
marks = m.code(["Hello, earthling."])          # LongTensor (1, 36): mark id per slot

# better: the writer rereads before sending (81% vs 33% exact on unseen natural text)
idx = SR.Searcher(m, rounds=20)(["Hello, earthling."])["idx"]   # symbols
marks = m.fsq_to_mark[idx]                                      # mark ids (1, 36)

img = H.export_image(m, marks)   # numpy ink in [0, 1], 1152 x 1152 (4x the 288 px render)
H.save_png("ring.png", img)      # 16-bit grayscale PNG, dark ink on white
```

On a laptop, keep the search batch small (`HEPTAPOD_CHUNK=32`). The browser version
([web/](../web/README.md)) produces the same mark ids client-side if you would rather not run Python.

## Checking a drawing

```bash
HEPTAPOD_OUT=v4/runs/s2 python v4/heptapod_t5.py read path/to/drawing.png
```

The reader expects dark ink on a light background, a square image with the ring centred, and the
same relative ring size as `export_image`. It downsamples to 288 px; exact 4x pixel blocks read back
identically. It was trained on clean renders plus slight rotation (about 4 degrees), scale (4%),
shift (3%) and noise. Smoky, stylised or photographed drawings are untested and will probably need
the eyes retrained on examples of that style.

## What the slots mean (measured, approximate)

The model was never told any of this; it is what the probes found
([results.md](results.md#what-the-slots-do)).

| Zone | Slots |
| --- | --- |
| Who (subject), written four times | 7, 9, 29, 35 |
| What it does (verb, tense, negation, agreement) | 2, 8, 13, 17, 25, 26 |
| On what (object noun, 2-4 marks per noun) | 14, 15, 10, 30 (+0, 4) |
| How it ends (statement or question, punctuation) | 22, 32, 33 |
| When / where (time, place, numbers) | 15, 19, 28 |

Slot 26 alone flips "The teacher is" to "The teachers are". The remaining slots carry spread-out
content. If your renderer animates the writing, these zones are a reasonable thing to highlight, but
the code is distributed: no slot is a word.

## Limits to keep in mind

Up to 12 English words. About 81% of unseen natural sentences come back exactly with rereading (94%
for 3-5 words), 75% with image noise. It fails on rare names, technical words, accents and nonsense
word lists.

## Attribution

Keep this with any rings you publish:

> Ink derived from the Arrival (2016) logograms by Martine Bertrand, released by Wolfram Research
> under CC BY-NC 4.0; marks cut and modified.

Non-commercial use only. See [NOTICE.md](../NOTICE.md).
