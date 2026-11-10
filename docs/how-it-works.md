# How it works

This page describes v4, the current system. The code is in [v4/heptapod_t5.py](../v4/heptapod_t5.py)
(model and training), [v4/search.py](../v4/search.py) (rereading) and
[v2/heptapod_ink.py](../v2/heptapod_ink.py) (the ink renderer).

## The ink

The film's logograms come from Wolfram Research's
[Arrival-Movie-Live-Coding](https://github.com/WolframResearch/Arrival-Movie-Live-Coding)
repository (38 logograms, CC BY-NC 4.0). Two scripts in [reference/](../reference/) turn them
into a library:

- `prep_library.py` fits a circle to each logogram (RANSAC) and unwraps the ring into a polar
  strip, so every ring sits on the same row.
- `prep_marks.py` cuts those strips into 83 whole marks (blots with their tendrils and droplets,
  never sliced through) and 341 clean pieces of ring line.

The result is `reference/marks.npz`. Nothing is drawn by hand and nothing is generated: every
pixel of ink in a ring comes from the film.

## Ring geometry

- 36 slots around a fixed ring of radius 0.5 on a [-1, 1] canvas, rendered at 288 x 288 px.
- Slot `k` is centred at `(k + 0.5) * 10` degrees, measured in image coordinates from 9 o'clock,
  going clockwise. The exact definition is in [integration.md](integration.md).
- Each slot holds one mark id, 0 meaning blank. Marks are squeezed along the ring to about 30
  degrees wide (`MARK_SCALE = 0.25`), so each one overlaps its neighbours and the result looks
  like a continuous logogram, not a row of glyphs.
- The ring line is the same piece in every logogram and carries no information.
- 81 marks are in use, so one ring can hold at most 36 x log2(81) = 228 bits.

## The models

**Writer.** The sentence goes through the T5-small encoder (fine-tuned at a low learning rate).
A slot writer with 36 learned queries attends over the encoder states (2-layer transformer
decoder, 256 dimensions) and produces one vector per slot. Each slot has its own codebook of 81
vectors (vector quantization, VQ); the nearest entry is the slot's symbol, and each symbol maps
to one ink mark. The codebooks are initialised by k-means after a continuous warm-up, updated by
exponential moving average, with a commitment loss, reseeding of dead entries and a per-slot
entropy term that keeps all 81 entries in use.

**Eyes.** The ring image is unwrapped into a polar strip (angle across, radius 0.20-0.86 down,
48 x 576 samples). A small CNN with wrap-around convolutions turns it into 72 angular tokens, a
2-layer transformer mixes them, and 36 slot queries output a probability over marks for each slot.
The eyes recognise about 99% of the marks on the writer's rings.

**Reader.** The eyes' probabilities (or, "on paper", the exact symbols) become a weighted mix of
codebook vectors. An adapter (2-layer transformer encoder, projection to 512, layer norm) turns
the 36 slot vectors into the memory that a pretrained T5-small decoder attends to. The decoder
writes the sentence.

## Training stages

Training the writer straight through the image did not work well (see [history.md](history.md)),
so v4 learns in stages:

0. **Eyes on random rings.** The eyes learn to recognise marks on random rings. Their real
   confusion matrix is measured and the marks they read worst are dropped, leaving 81.
1. **On paper.** Writer and reader invent the code with no image in between. Part of the symbols
   go through the measured confusion matrix, so the code learns to survive the eyes' real mistakes.
2. **Learning to see.** The writer is frozen. The eyes learn to read its rings (the slot labels
   are free, since we know what was drawn), then learn to produce text through the frozen reader.
   Late in this stage the decoder adapts gently.

During training the image gets slight rotation (about 4 degrees), scale (4%), shift (3%) and
noise. Orientation is fixed, as in the film.

**Data.** The first version trained on Tatoeba English sentences of up to 12 words (1.88M),
mixed with random word combinations from a 32k vocabulary (50% in stage 1, 30% in stage 2) to
force the code to carry information the decoder cannot guess. The current checkpoint
(`v4/runs/s2/model.pt`) continued stage 1 for 40 minutes on a 500k-sentence mix of natural text
(FineWeb 55%, Wikipedia 15%, CC-News 10%, Tatoeba 20%, sentences of 2-12 words, deduplicated
against every evaluation set), then adapted the eyes for 15 minutes. The mix is built by
[v4/prep_mixed.py](../v4/prep_mixed.py). Training ran on a rented H100 (Modal).

## Rereading (sender-side search)

The plain writer picks the nearest codebook entry per slot. That loses information: continuous
slots reach 99% exact, the quantized code much less. Rereading recovers a lot of it without
changing any weights:

1. Start from the writer's own ring.
2. The gradient of the reader's loss with respect to the symbols ranks single-slot swaps; for each
   slot the candidates are the 16 nearest codebook entries.
3. The best proposals are scored exactly on the symbolic path, and the finalists are rendered
   and read through the eyes on a fixed set of views (clean plus seeded perturbations).
4. A swap is accepted only if it lowers the reading loss. Up to 20 rounds.
5. At the end, the new ring is kept only if its clean reading is not worse than the original's.

Only trained models take part: the frozen writer proposes, a frozen copy of the reader judges.
The receiver still gets nothing but pixels. On the locked exam this takes about 0.7 seconds per
sentence on an H100 and accepts about 13 swaps on average. The browser version uses a faster
variant (8 alternatives per slot, several moves per round, early stop when the reading is exact),
described in [web/README.md](../web/README.md).

## Browser version

[web/](../web/README.md) runs the same model with onnxruntime-web: the writer, renderer, eyes and
decoder are exported to ONNX (int8 or fp16, about 168 MB in total) and the T5 tokenizer comes from
transformers.js. A first ring is drawn in about 20-30 ms; rereading takes a few seconds.
