# Heptapod Ink v3: state of play (2026-11-09)

## Goal (fixed)
A model that "speaks Heptapod": text -> one ring logogram written with real ink from the film Arrival -> text. Trained end to end; nothing about grammar or meaning hand-coded. Success = >= 90% exact-match reconstruction of sentences never seen in training (COGS IID test/dev), then a fresh reader learning the frozen code. Human readability is out of scope.

## Architecture (v3/heptapod.py)
- Input: frozen all-MiniLM-L6-v2, contextual token vectors (384-d).
- Writer: 2-layer transformer encoder over tokens + 24 learned slot queries (2-layer transformer decoder, cross-attention) -> per slot logits over 33 ring segments (incl. blank) and 84 whole ink marks (incl. none).
- Discretization: hand-written straight-through Gumbel-softmax. Soft (mixture) phase for the first 40% of steps, then hard. tau = 2 * 0.25**frac.
- Script renderer (v2/heptapod_ink.py RingMarkRenderer): 24 slots at 15 deg pitch, ring radius 0.5 on a 192x192 canvas; ring segments feathered, marks pasted whole (union 1-prod(1-x)), marks condensed along the ring to 40% or 25% of their original angular width. Marks are real ink cut from the 38 Arrival logograms (83 marks, median 47 deg wide before condensing).
- Channel (train only): +-4 deg rotation, +-4% scale, +-3% shift, Gaussian noise 0.1. Fixed orientation (canon).
- Reader: "polar eyes" (grid_sample unwrap of radius 0.20-0.86 to a 48x384 strip, circular-padded 3x3 convs, pooled to 48 angular tokens) + 2-layer transformer encoder + 3-layer transformer decoder over BERT WordPiece vocab, greedy decoding.
- Losses: token CE with 30% word dropout on decoder inputs + bag-of-words BCE from the image (x100) + ink ceiling penalty (ink > 12%).
- Optim: AdamW lr 5e-4, OneCycle, batch 64, grad clip 1.0. H100 on Modal: ~0.06 s/step.
- Data: COGS train deduplicated by canonical meaning graph (23,657 sentences), dev 2,862, test 2,858, gen 20,974 (all deduped against train).

## Results
| run | config | best dev exact (512 sents) | final test exact | notes |
|---|---|---|---|---|
| base | marks 40%, 30k steps | 43.8% @30k | 44.5% (train 61.9%) | dip at hard switch (17%) then steady climb, still rising at end |
| narrow | marks 25%, 30k | 39.3% at the switch, then 4-25% | 35.0% (best ckpt) | unstable after hard switch |
| long | marks 40%, 80k (soft phase 32k) | 32.2% at switch | killed | collapsed: ink fell to 0.4% (writer stopped writing), info gap halved, 2.5% |
| freeze25 | marks 25%, writer frozen at switch (argmax code), reader keeps training (lr*0.5, cosine) | 33.4% at switch -> 63.1% (+2k) -> 66.4% -> 65.6% | running | word accuracy 90.6% at 63% exact |
| freeze40 | marks 40%, same freeze | 15.2% -> 59.4% -> 63.7% -> 64.5% | running | word acc 87% at 59% |
COGS gen categories: ~0% for every run so far.

## Diagnostics we have
- Random-code capacity probe (reader trained on random slot choices, 1200 steps): marks recognised when present 64% (24 slots, marks 40%) -> 71-79% (marks condensed to 40%/25%); ring segments 15-42%. Before polar eyes: ~chance.
- Information check (decoder CE with the right logogram vs a shuffled one): base end 0.30 vs 7.95; freeze runs ~0.2-0.35 vs ~4.
- Writer uses ~2-3% ink in hard phase (~20-30% of slots marked).
- Nominal capacity: 24 x log2(33*84) ~ 274 bits; COGS sentences need maybe ~60-100 bits.

## Constraints
- Budget left this month on Modal: ~USD 15 (H100 ~USD 3.95/h, ~0.06 s/step => ~16k steps per USD). M4 Pro laptop also available (8x slower).
- Keep the film's ink and the ring (the script). Everything else (architecture, losses, training schedule, slots count, mark scale) is open.

## Files
v3/heptapod.py, v2/heptapod_ink.py, v3/channel_probe.py, research/notes/*.md (codex_methodology.md, arrival_sources.md, drawing_games.md, meanings_and_survey.md), research/SYNTHESIS.md.
