# Heptapod Ink v2: an emergent script written with the film's own ink

Same game as v1 (a speaker writes one ring per whole sentence, a listener reads it at a random rotation), but the brush is made of **real ink from Arrival**.

## Ink library

`reference/` holds the 38 logograms from the film (by Martine Bertrand), published by Wolfram Research in [WolframResearch/Arrival-Movie-Live-Coding](https://github.com/WolframResearch/Arrival-Movie-Live-Coding) under CC BY-NC 4.0. Everything derived from them here is non-commercial too; see [NOTICE.md](../NOTICE.md).

- `reference/prep_library.py` fits each ring (RANSAC circle) and unwraps it into a polar strip, so every ring sits on the same row.
- `reference/prep_marks.py` cuts the strips into **83 whole marks** (clusters with their tendrils and droplets, never sliced) and **341 clean ring segments**. The model uses 40 of the segments, spread from hairline to heavy.

A logogram is 12 slots around the ring. For each slot the speaker picks a ring segment (or a gap) and a mark (or none). Marks are pasted whole and may spill into neighbouring slots.

## Run

```bash
python v2/heptapod_ink.py train         # 8000 steps, ~45 min on an M4 Pro
python v2/heptapod_ink.py say heptapod offers weapon --tense future
python v2/heptapod_ink.py poster
```

## What it took to make it learn

| Run | Change | Whole sentence (held out) | Subject |
| --- | --- | --- | --- |
| `out_run1` | 341 ring segments, hard choices from the start | 1% | chance |
| `out_run2` | 40 ring segments, longer training | 0% | chance (stuck at log 6) |
| `out_run3` | soft mixtures for the first 40% of training, extra loss weight for lagging slots | **94%** | 96% |
| `out_run4` (also copied to `out`) | 32 segments, plus a target of about 3.5 marks per logogram to thin the ink | 57% | 79% |

Note that run 3 and run 4 are different configurations (40 vs 32 ring segments, no mark target vs a mark target), so their numbers should not be mixed. The current defaults in `heptapod_ink.py` match run 4. Discrete choices are much harder to learn than v1's continuous brush. Without the soft start, one word (the subject) got abandoned in a local optimum and never recovered.

## Results (`out_run3/`, the best run)

- Training sentences: 98.8% read exactly, fully rotated and noisy
- Held-out subject-verb-object triples: 93.9%
- Topographic similarity 0.42 (shuffled 0), which suggests a compositional script. A later review found flaws in how this was measured (see [research/SYNTHESIS.md](../research/SYNTHESIS.md))
- It uses 22 ring segments and 34 of the 83 marks, about 10 marks per logogram, with gaps in 26% of the slots

## Known artifacts

- Heavy ring segments are clipped to the ring band, which can leave small square-ended blocks. Using only thin and medium segments, or fading the band radially, would fix it.
- About 10 marks per logogram is busier than most film logograms. Raising the ink weight, or adding a per-mark cost, should thin it out.

`heptapod_ink.py` also holds the ring renderer (`RingMarkRenderer`) that v3 and v4 reuse.
