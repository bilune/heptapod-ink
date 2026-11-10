# Heptapod Ink: next phase = read and write ANY English sentence

## Goal
A trained system that writes any English sentence (<= 12 words) as ONE ring logogram made of real ink from the film Arrival, and reads it back to English. Fully learned (no hand-coded grammar/meaning). Must leverage existing pretrained models where possible. The ink and the ring are the fixed "script": a ring with N slots; each slot holds one of ~33 ring segments (incl. gap) and one of ~84 whole ink marks (incl. none), rendered by v2/heptapod_ink.py RingMarkRenderer (marks can be condensed along the ring), fixed orientation like the film. One ring per utterance; if needed, multiple rings per sentence is allowed (film-faithful).

## What worked on COGS (v3, see v3/README.md and v3/heptapod.py)
- Writer: frozen all-MiniLM-L6-v2 token vectors -> 2-layer transformer + 24 slot queries -> per-slot logits (ring, mark). Straight-through Gumbel, soft (mixture) phase for 40% of steps, then FREEZE the writer (argmax code) and train only the reader. Freezing jumped dev exact 33% -> 63-71%.
- Reader: "polar eyes" (unwrap ring to 48x384 strip, circular convs, 48 angular tokens) + 2-layer encoder + 3-layer decoder from scratch over BERT WordPiece vocab. Plain CNN could not read the ring at all.
- Train-only lexical augmentation for the reader: 71% -> 76-77.5%. A brand-new reader trained from scratch on the frozen code reached 78.3% on test: the code is learnable.
- Read-and-verify: beam 32 candidates, frozen writer re-renders each, pick the matching drawing -> 93.2% exact on COGS test (vs 78.3% reader alone). 0 code collisions.
- Failure mode: rare words (93% error for words seen 1-9 times). COGS gen (structural) ~0%.
- Arbitrary English through the COGS model fails completely (it outputs COGS-like sentences).

## Capacity probe (random slot codes, reader trained 1.5k steps, marks condensed to 25%)
24 slots: present-mark recognition 79% (~80 bits est.); 36 slots: 71% (~105 bits); 48 slots: 50% (~105 bits); 48 slots at 288px: 51% (but polar eyes sample a fixed 384x48 strip, so resolution was not really tested). Overlap of marks limits beyond ~36 slots.

## Data ready
v3/data_tatoeba/: 1,878,790 train, 3,000 dev, 3,000 test Tatoeba English sentences (<=12 words, deduped, no [UNK] for the MiniLM WordPiece tokenizer), plus 1,000 dev and 1,000 test random word combinations (3-10 words) and random_vocab.json.

## Budget and compute
~USD 16 of Modal credit left this month (H100 ~USD 3.95/h, billed per second). Apple M4 Pro 24GB laptop also available (MPS, ~8x slower than H100). Python env: torch, transformers, sentence-transformers available; anything on Hugging Face can be downloaded.

## Options Claude is considering
A. Same v3 pipeline, but the reader's decoder is a pretrained T5-small decoder (fed with polar-eye tokens as encoder_outputs), MiniLM fine-tuned with low LR instead of frozen, 36 slots.
B. "MiniLM becomes the writer": append 36 learned [SLOT] tokens to the sentence, run (fine-tuned) MiniLM, each slot token's output -> linear heads for (ring, mark). No separate writer transformer. Reader = T5 converted into a heptapod reader (polar eyes -> T5 encoder or directly as encoder_outputs -> T5 decoder).
