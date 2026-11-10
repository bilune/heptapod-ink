# Distillation of sender search into the writer fails: find the bug or the reason

Read v4/distill.py (train()), v4/search.py (Searcher.exhaustive), v4/heptapod_t5.py (SlotWriter, writer_out, encode, books).
Targets: 8,704 train sentences (50% natural, 50% random) with searched symbol indices (36 slots, 81 per-slot VQ codebook entries, 256-d, frozen books). At init the writer's own nearest-centroid code agrees with the targets on 83% of slots (search changes ~6 slots). Raw writer vector norm ~66, book entry norm ~76; mean squared distance raw->own nearest centroid is 0.44x mean book energy (raw vectors are far from centroids; VQ commitment was 0.25 during v4 training).

Attempts (writer only trainable unless stated, eval mode, Adam):
1. H100: CE on -mean_D(dist)/tau_s with tau_s calibrated to entropy ln4 (tiny tau) + text CE through hard STE, writer lr 3e-4, encoder 2e-5: slot CE exploded 190 -> 250,000, raw norm blew up, writer destroyed (0% on dev).
2. M4: relative MSE(raw, goal centroid)/mean(books^2) + text CE (weight 1): raw norm grew 66 -> 200-500, target match fell 0.83 -> ~0.6, dev panels flat/noisy.
3. M4: MSE(raw, goal) + 0.25 commit(fresh) + 0.1 text, lr 1e-4, encoder frozen: raw norm 66 -> 45, target match 0.83 -> 0.33-0.44, dev collapsed.
4. M4 single fixed batch of 32, pure MSE(raw, goal), lr 1e-5, 30 steps: MSE 12.7 -> 9.95 but match 0.83 -> 0.75.
5. Same with CE(-d/std(d)) + 0.1 MSE, lr 1e-5 and 1e-4, 60 steps: match on the training batch falls 0.83 -> 0.72-0.74; CE barely moves (2.82 -> 2.84 at 1e-5).
Forward is deterministic (no-grad vs grad differ by 4e-4).

Questions: why does fitting a fixed batch reduce argmin agreement with targets? Is there a bug in the loss/indexing (e.g. books indexing [arange(S)[None], target] with target (B,S)), a geometry issue (decision regions vs centroids when raw is far from all centroids), or a writer-architecture issue (slot queries share a decoder; one gradient moves all slots)? Propose the correct imitation objective and a 2-minute local test that must pass before spending GPU. Also say plainly if distillation is not worth it versus keeping search at send time (~1 s/sentence on H100). <=500 words. Write research/debate3/round3_codex.md.
