# Debate 3: how to raise v4's accuracy (open English in one Heptapod ring)

Read first: v4/README.md, v4/heptapod_t5.py, v4/check_v4.py, research/debate2/round1_codex.md, round2_codex.md, round3_codex.md.

## The system (v4, final)
text -> T5-small encoder (fine-tuned) -> SlotWriter (36 learned queries, 2-layer transformer decoder over encoder states, 256-d) -> per-slot VQ: 36 codebooks x 81 entries x 256-d (k-means init after a 4,000-step continuous warm-up; EMA .99; commitment .25; dead-entry reseed; per-slot batch-entropy term lambda .1) -> each slot = one of 81 real film-ink marks on a fixed ring (36 slots, 10 deg pitch, marks 30 deg wide so they overlap neighbours; render 288 px) -> eyes (polar unwrap 48x576, circular CNN, 72 angular tokens, 2-layer encoder, 36 slot queries) -> per-slot mark distribution -> expected codebook vector -> adapter (2-layer encoder, Linear->512, LayerNorm) -> pretrained T5-small decoder -> text.
Stages: 0 eyes on random rings (91% per mark), keep 81 most readable marks + confusion matrix; 1 "on paper" (no image), symbols sent through the measured confusion channel (alpha 1); 2 writer frozen, eyes trained with slot CE then text CE through the frozen symbolic reader, decoder adapts at 1e-5 with 20% clean replay.
Data: Tatoeba English <=12 words (1.88M train), 50% fresh random word combinations from a 32k vocab during stage 1, 30% in stage 2. Held-out: dev/test 3,000 natural; dev_random/test_random 1,000 random combos.

## Results (test, run once)
- Natural: reads alone 62.7% exact / 88.2% words; read-and-verify (beam 32, writer redraws, pixel match) 88.2% exact / 94.1% words; gold in candidates 90.1%; false match 2.1%.
- Random combos: 9.2% exact / 44.1% words alone; verified 29.0% exact / 48.3% words; gold in candidates 30.0%.
- Shuffled-image control ~0. Eyes read 99.1% of marks (98.1% on random). Symbolic (no image) is only ~2 points above visual: vision is not the bottleneck.
- Continuous 256-d slots (no quantization) reached 99% exact on both natural and random. The loss is the continuous -> 81-symbol step.
- On natural dev, 22% of sentences have another beam candidate producing the identical ring (code collisions: near capacity).
- Errors on random combos keep word shape but swap rare words ("marika my rest what advocate mary" -> "maryland my stay what suggest mary").
- Nominal capacity 36 x log2(81) = 228 bits; a random 10-word combo from 32k vocab needs ~150 bits; actual usage per slot is uneven (some slots used 10-12 marks only).

## History of what failed (do not re-propose without a new reason)
categorical 82-way argmax/Gumbel STE (info lived in soft probs); FSQ 4x{-1,0,1} and continuous 4-number control (too narrow, did not learn); shared random-init VQ (collapsed); a Modal run hung silently once.

## Constraints
- Trained models only; no hand-coded grammar or word->mark tables. Human readability out of scope. One ring per sentence is the film's convention (constant ring size; complexity = more ink). Changing geometry is allowed if justified, but say so.
- Budget: Gonzalo's own Modal H100 (~USD 3.95/h). The previous goal cap is spent; a new budget must be justified and proposed explicitly (e.g. 5 / 10 / 20 USD tiers).
- Hardware locally: M4 Pro 24GB (MPS) for smoke tests and evaluation.
- Every change smoke-tested locally (incl. bf16 path) before the H100.

## The question
What is the best way to raise accuracy, especially on random word combinations (the real test that the ring carries words, not gist), and exact match on natural text? Think broadly and out of the box.
